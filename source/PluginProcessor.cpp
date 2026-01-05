#include "PluginProcessor.h"

#include <array>
#include <chrono>
#include <cstring>

#include "PluginEditor.h"

namespace {
constexpr auto kSoftClipperKernel = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct DistortionParams
{
    float drive;
    float mix;
    float outputGain;
};

kernel void soft_clipper(
    const device float* inBuffer [[buffer(0)]],
    device float* outBuffer [[buffer(1)]],
    constant DistortionParams& params [[buffer(2)]],
    constant uint& bufferSize [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    // Bounds check to prevent buffer overruns
    if (gid >= bufferSize)
        return;

    float input = inBuffer[gid];
    float driven = input * params.drive;
    float clipped = tanh(driven);
    float processed = clipped * params.outputGain;
    float output = input * (1.0f - params.mix) + processed * params.mix;
    outBuffer[gid] = output;
}
)METAL";
}  // namespace

juce::AudioProcessorValueTreeState::ParameterLayout
PluginProcessor::createParameterLayout() {
  std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;
  // Parameters will be added when waveguide engine is integrated
  return {params.begin(), params.end()};
}

//==============================================================================
PluginProcessor::PluginProcessor()
    : AudioProcessor(
          BusesProperties()
#if !JucePlugin_IsMidiEffect
#if !JucePlugin_IsSynth
              .withInput("Input", juce::AudioChannelSet::stereo(), true)
#endif
              .withOutput("Output", juce::AudioChannelSet::stereo(), true)
#endif
              ),
      parameters(*this, nullptr, "Parameters", createParameterLayout()) {
}

PluginProcessor::~PluginProcessor() {
}

//==============================================================================
const juce::String PluginProcessor::getName() const { return JucePlugin_Name; }

bool PluginProcessor::acceptsMidi() const {
#if JucePlugin_WantsMidiInput
  return true;
#else
  return false;
#endif
}

bool PluginProcessor::producesMidi() const {
#if JucePlugin_ProducesMidiOutput
  return true;
#else
  return false;
#endif
}

bool PluginProcessor::isMidiEffect() const {
#if JucePlugin_IsMidiEffect
  return true;
#else
  return false;
#endif
}

double PluginProcessor::getTailLengthSeconds() const {
  if (asyncActive.load() && lastSampleRate > 0.0)
    return static_cast<double>(lastBlockSize) / lastSampleRate;

  return 0.0;
}

int PluginProcessor::getNumPrograms() {
  return 1;  // NB: some hosts don't cope very well if you tell them there are 0
             // programs,
  // so this should be at least 1, even if you're not really implementing
  // programs.
}

int PluginProcessor::getCurrentProgram() { return 0; }

void PluginProcessor::setCurrentProgram(int index) {
  juce::ignoreUnused(index);
}

const juce::String PluginProcessor::getProgramName(int index) {
  juce::ignoreUnused(index);
  return {};
}

void PluginProcessor::changeProgramName(int index,
                                        const juce::String& newName) {
  juce::ignoreUnused(index, newName);
}

//==============================================================================
void PluginProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) {
  lastSampleRate = sampleRate;
  lastBlockSize = samplesPerBlock;

  const int totalChannels =
      juce::jmax(getTotalNumInputChannels(), getTotalNumOutputChannels());

  asyncConfig.maxChannels = juce::jmax(1, totalChannels);
  asyncConfig.maxSamplesPerBlock = juce::jmax(1, samplesPerBlock);
  asyncConfig.queueDepth = 3;
  asyncConfig.maxParameterBytes = sizeof(DistortionParams);

  if (auto initResult = accelerator.initialize(); !initResult) {
    auto error = initResult.error();
    updateStatus("Metal init failed: " + juce::String(error.message));
    return;
  }

  if (auto loadResult = accelerator.loadKernel(kernelName, kSoftClipperKernel);
      !loadResult) {
    auto error = loadResult.error();
    updateStatus("Kernel load failed: " + juce::String(error.message));
    return;
  }

  applyAsyncConfiguration();

  // Initialize waveguide engine
  ts::metal::waveguide::WaveguideConfig wgConfig;
  wgConfig.maxWaveguides = 50;
  wgConfig.sampleRate = static_cast<float>(sampleRate);
  wgConfig.maxDelayLineSamples = 1200;

  std::string wgError;
  if (waveguideEngine.prepare(wgConfig, wgError)) {
    const juce::SpinLock::ScopedLockType lock(waveguideStatusLock);
    waveguideStatus = juce::String(waveguideEngine.statusString());
  } else {
    const juce::SpinLock::ScopedLockType lock(waveguideStatusLock);
    waveguideStatus = "Waveguide error: " + juce::String(wgError);
    waveguideEnabled.store(false);
  }

  if (asyncActive.load())
    setLatencySamples(samplesPerBlock);
  else
    setLatencySamples(0);
}

void PluginProcessor::releaseResources() {
  accelerator.shutdown();
  asyncActive.store(false);
  waveguideEngine.shutdown();
}

bool PluginProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
#if JucePlugin_IsMidiEffect
  juce::ignoreUnused(layouts);
  return true;
#else
  // This is the place where you check if the layout is supported.
  // In this template code we only support mono or stereo.
  if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono() &&
      layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
    return false;

// This checks if the input layout matches the output layout
#if !JucePlugin_IsSynth
  if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
    return false;
#endif

  return true;
#endif
}

void PluginProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                   juce::MidiBuffer& midiMessages) {
  juce::ignoreUnused(midiMessages);

  juce::ScopedNoDenormals noDenormals;

  if (asyncConfigPending.load(std::memory_order_acquire)) {
    asyncConfigPending.store(false, std::memory_order_release);
    applyAsyncConfiguration();
  }

  auto totalNumInputChannels = getTotalNumInputChannels();
  auto totalNumOutputChannels = getTotalNumOutputChannels();

  for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
    buffer.clear(i, 0, buffer.getNumSamples());

  // Waveguide processing
  if (waveguideEnabled.load() && waveguideEngine.isInitialized()) {
    const float* inputPtrs[2]{
        totalNumInputChannels > 0 ? buffer.getReadPointer(0) : nullptr,
        totalNumInputChannels > 1 ? buffer.getReadPointer(1) : nullptr};

    float* outputPtrs[2]{buffer.getWritePointer(0),
                         totalNumOutputChannels > 1 ? buffer.getWritePointer(1)
                                                    : buffer.getWritePointer(0)};

    waveguideEngine.processBlock(inputPtrs, totalNumInputChannels, outputPtrs,
                                 totalNumOutputChannels, buffer.getNumSamples());

    for (int ch = 2; ch < totalNumOutputChannels; ++ch)
      buffer.clear(ch, 0, buffer.getNumSamples());

    return;
  }

  // Fallback to soft clipper demo if waveguide disabled
  const DistortionParams params = distortionParams;

  if (asyncActive.load() && accelerator.isAsyncEnabled()) {
    auto result = accelerator.processBlockAsync(kernelName, buffer, &params,
                                                sizeof(params));
    if (!result) {
      auto error = result.error();
      asyncActive.store(false);
      asyncRequested.store(false);
      updateStatus("Async error: " + juce::String(error.message) +
                   " (fallback to sync)");

      requestAsyncTeardown();

      auto syncResult =
          processSynchronously(kernelName, buffer, &params, sizeof(params));
      if (!syncResult) {
        updateStatus("Sync error: " + juce::String(syncResult.error().message));
        buffer.clear();
      }
      return;
    }
    return;
  }

  auto syncResult =
      processSynchronously(kernelName, buffer, &params, sizeof(params));
  if (!syncResult) {
    updateStatus("Sync error: " + juce::String(syncResult.error().message));
    buffer.clear();
  }
}

//==============================================================================
bool PluginProcessor::hasEditor() const {
  return true;  // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor* PluginProcessor::createEditor() {
  return new PluginEditor(*this);
}

//==============================================================================
void PluginProcessor::getStateInformation(juce::MemoryBlock& destData) {
  auto state = parameters.copyState();
  state.setProperty("asyncRequested", asyncRequested.load(), nullptr);

  if (auto xml = state.createXml()) copyXmlToBinary(*xml, destData);
}

void PluginProcessor::setStateInformation(const void* data, int sizeInBytes) {
  if (auto xml = getXmlFromBinary(data, sizeInBytes)) {
    auto state = juce::ValueTree::fromXml(*xml);
    if (state.isValid()) {
      const bool asyncMode = static_cast<bool>(
          state.getProperty("asyncRequested", asyncRequested.load()));

      parameters.replaceState(state);
      setProcessingModeAsync(asyncMode);
    }
  }
}

void PluginProcessor::setProcessingModeAsync(bool useAsync) {
  const bool previous = asyncRequested.exchange(useAsync);
  if (previous == useAsync) return;

  if (!accelerator.isInitialized()) return;

  // Signal the audio thread to apply configuration change at a safe point
  // rather than forcing suspension which can race with processBlock
  asyncConfigPending.store(true, std::memory_order_release);
}

bool PluginProcessor::isAsyncRequested() const { return asyncRequested.load(); }

bool PluginProcessor::isAsyncActive() const { return asyncActive.load(); }

juce::String PluginProcessor::getProcessingStatus() const {
  const juce::SpinLock::ScopedLockType lock(statusLock);
  return statusText;
}

ts::metal::MetalAccelerator::GPUStats PluginProcessor::getGPUStats() const {
  return accelerator.getGPUStats();
}

void PluginProcessor::setWaveguideEnabled(bool enabled) {
  waveguideEnabled.store(enabled);
}

bool PluginProcessor::isWaveguideEnabled() const {
  return waveguideEnabled.load();
}

juce::String PluginProcessor::getWaveguideStatus() const {
  const juce::SpinLock::ScopedLockType lock(waveguideStatusLock);
  return waveguideStatus;
}

int PluginProcessor::getWaveguideCount() const {
  return waveguideEngine.getWaveguideCount();
}

void PluginProcessor::applyAsyncConfiguration() {
  const bool wantAsync = asyncRequested.load();

  if (!wantAsync) {
    if (accelerator.isAsyncEnabled()) requestAsyncTeardown();

    asyncActive.store(false);
    setLatencySamples(0);  // No latency in sync mode
    updateStatus("Phase 1 (synchronous)");
    return;
  }

  auto result = accelerator.enableAsyncProcessing(asyncConfig);
  if (!result) {
    asyncActive.store(false);
    setLatencySamples(0);
    updateStatus("Async unavailable: " + juce::String(result.error().message));
    return;
  }

  asyncActive.store(true);
  setLatencySamples(lastBlockSize);  // One block of latency in async mode
  updateStatus("Phase 2 (async GPU)");
}

void PluginProcessor::updateStatus(juce::String newStatus) {
  const juce::SpinLock::ScopedLockType lock(statusLock);
  statusText = newStatus;
}

std::expected<void, ts::metal::MetalErrorInfo>
PluginProcessor::processSynchronously(const std::string& name,
                                      juce::AudioBuffer<float>& buffer,
                                      const void* params, size_t paramsSize) {
  return accelerator.processBlock(name, buffer, params, paramsSize);
}

void PluginProcessor::handleAsyncUpdate() {
  if (asyncTeardownPending.exchange(false))
    accelerator.disableAsyncProcessing();
}

void PluginProcessor::requestAsyncTeardown() {
  const bool wasPending = asyncTeardownPending.exchange(true);
  if (!wasPending) triggerAsyncUpdate();
}

void PluginProcessor::parameterChanged(const juce::String& parameterID,
                                       float newValue) {
  juce::ignoreUnused(parameterID, newValue);
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
  return new PluginProcessor();
}
