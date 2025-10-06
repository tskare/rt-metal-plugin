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

  params.emplace_back(std::make_unique<juce::AudioParameterChoice>(
      "room_size", "Room Size", juce::StringArray{"Small", "Medium", "Large"},
      1));

  params.emplace_back(std::make_unique<juce::AudioParameterFloat>(
      "fdtd_absorption", "Absorption",
      juce::NormalisableRange<float>{0.90f, 0.999f, 0.0f, 0.5f}, 0.995f));

  params.emplace_back(std::make_unique<juce::AudioParameterFloat>(
      "fdtd_wet", "Wet Level",
      juce::NormalisableRange<float>{0.0f, 1.0f, 0.0f, 1.0f}, 1.0f));

  params.emplace_back(std::make_unique<juce::AudioParameterFloat>(
      "fdtd_dry", "Dry Level",
      juce::NormalisableRange<float>{0.0f, 1.0f, 0.0f, 1.0f}, 0.2f));

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
  parameters.addParameterListener("room_size", this);
  parameters.addParameterListener("fdtd_absorption", this);
  parameters.addParameterListener("fdtd_wet", this);
  parameters.addParameterListener("fdtd_dry", this);
}

PluginProcessor::~PluginProcessor() {
  parameters.removeParameterListener("room_size", this);
  parameters.removeParameterListener("fdtd_absorption", this);
  parameters.removeParameterListener("fdtd_wet", this);
  parameters.removeParameterListener("fdtd_dry", this);
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

  if (fdtdEnabled.load() && fdtdEngine.usesGPU() && lastSampleRate > 0.0)
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

  configureFDTD(sampleRate);
  fdtdEngine.setExpectedBlockSize(samplesPerBlock);
  fdtdEngine.setWatchdogTimeout(std::chrono::nanoseconds::zero());

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

  // Update latency compensation (one block delay for async GPU or FDTD)
  if (asyncActive.load() || fdtdEnabled.load())
    setLatencySamples(samplesPerBlock);
  else
    setLatencySamples(0);
}

void PluginProcessor::releaseResources() {
  accelerator.shutdown();
  asyncActive.store(false);
  fdtdEngine.shutdown();
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

  // Check if configuration change is pending from UI thread
  // Apply it here in the audio thread at a safe point
  if (asyncConfigPending.load(std::memory_order_acquire)) {
    asyncConfigPending.store(false, std::memory_order_release);
    applyAsyncConfiguration();
  }

  auto totalNumInputChannels = getTotalNumInputChannels();
  auto totalNumOutputChannels = getTotalNumOutputChannels();

  // In case we have more outputs than inputs, this code clears any output
  // channels that didn't contain input data, (because these aren't
  // guaranteed to be empty - they may contain garbage).
  // This is here to avoid people getting screaming feedback
  // when they first compile a plugin, but obviously you don't need to keep
  // this code if your algorithm always overwrites all the output channels.
  for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
    buffer.clear(i, 0, buffer.getNumSamples());

  if (fdtdEnabled.load()) {
    const float* inputPtrs[2]{
        totalNumInputChannels > 0 ? buffer.getReadPointer(0) : nullptr,
        totalNumInputChannels > 1 ? buffer.getReadPointer(1) : nullptr};

    float* outputPtrs[2]{buffer.getWritePointer(0),
                         totalNumOutputChannels > 1
                             ? buffer.getWritePointer(1)
                             : buffer.getWritePointer(0)};

    fdtdEngine.processBlock(inputPtrs, totalNumInputChannels, outputPtrs,
                            totalNumOutputChannels, buffer.getNumSamples());

    for (int ch = 2; ch < totalNumOutputChannels; ++ch)
      buffer.clear(ch, 0, buffer.getNumSamples());

    return;
  }

  // This is the place where you'd normally do the guts of your plugin's
  // audio processing...
  // Make sure to reset the state if your inner loop is processing
  // the samples and the outer loop is handling the channels.
  // Alternatively, you can process the samples with the channels
  // interleaved by keeping the same state.
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
  state.setProperty("fdtdEnabled", fdtdEnabled.load(), nullptr);
  state.setProperty("fdtdGPUPreferred", fdtdGPUPreferred.load(), nullptr);

  if (auto xml = state.createXml()) copyXmlToBinary(*xml, destData);
}

void PluginProcessor::setStateInformation(const void* data, int sizeInBytes) {
  if (auto xml = getXmlFromBinary(data, sizeInBytes)) {
    auto state = juce::ValueTree::fromXml(*xml);
    if (state.isValid()) {
      const bool asyncMode = static_cast<bool>(
          state.getProperty("asyncRequested", asyncRequested.load()));
      const bool fdtdMode = static_cast<bool>(
          state.getProperty("fdtdEnabled", fdtdEnabled.load()));
      const bool fdtdGpuPref = static_cast<bool>(
          state.getProperty("fdtdGPUPreferred", fdtdGPUPreferred.load()));

      parameters.replaceState(state);
      setProcessingModeAsync(asyncMode);
      setFDTDEnabled(fdtdMode);
      setFDTDGPUPreferred(fdtdGpuPref);
      fdtdConfigDirty.store(true);
      triggerAsyncUpdate();
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

void PluginProcessor::setFDTDEnabled(bool enabled) {
  fdtdEnabled.store(enabled);

  // Update latency compensation when FDTD toggled
  if (enabled || asyncActive.load())
    setLatencySamples(lastBlockSize);
  else
    setLatencySamples(0);

  fdtdEngine.setGPUPreferred(fdtdGPUPreferred.load());

  const juce::SpinLock::ScopedLockType lock(fdtdStatusLock);
  fdtdStatus = enabled ? juce::String(fdtdEngine.statusString())
                       : juce::String("FDTD disabled");
}

bool PluginProcessor::isFDTDEnabled() const { return fdtdEnabled.load(); }

juce::String PluginProcessor::getFDTDStatus() const {
  const juce::SpinLock::ScopedLockType lock(fdtdStatusLock);
  return fdtdStatus;
}

void PluginProcessor::setFDTDGPUPreferred(bool useGPU) {
  const bool previous = fdtdGPUPreferred.exchange(useGPU);
  if (previous == useGPU) return;

  fdtdEngine.setGPUPreferred(useGPU);

  {
    const juce::SpinLock::ScopedLockType lock(fdtdStatusLock);
    fdtdStatus = juce::String(fdtdEngine.statusString());
  }

  if (fdtdEnabled.load()) {
    const bool gpuInUse = useGPU && fdtdEngine.usesGPU();
    if (gpuInUse || asyncActive.load())
      setLatencySamples(lastBlockSize);
    else
      setLatencySamples(0);
  }
}

bool PluginProcessor::isFDTDGPUPreferred() const {
  return fdtdGPUPreferred.load();
}

ts::metal::fdtd::FDTDReverbEngine::AsyncMetrics
PluginProcessor::getFDTDAsyncMetrics() const {
  return fdtdEngine.getAsyncMetrics();
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

  if (fdtdConfigDirty.exchange(false)) configureFDTD(lastSampleRate);
}

void PluginProcessor::requestAsyncTeardown() {
  const bool wasPending = asyncTeardownPending.exchange(true);
  if (!wasPending) triggerAsyncUpdate();
}

void PluginProcessor::parameterChanged(const juce::String& parameterID,
                                       float newValue) {
  juce::ignoreUnused(newValue);

  if (parameterID == "room_size" || parameterID == "fdtd_absorption" ||
      parameterID == "fdtd_wet" || parameterID == "fdtd_dry") {
    fdtdConfigDirty.store(true, std::memory_order_release);
    triggerAsyncUpdate();
  }
}

void PluginProcessor::configureFDTD(double sampleRate) {
  static constexpr struct {
    int nx, ny, nz;
  } roomSizes[]{{16, 14, 12}, {24, 20, 18}, {32, 26, 24}};

  const auto* sizeValue = parameters.getRawParameterValue("room_size");
  const auto* absorptionValue =
      parameters.getRawParameterValue("fdtd_absorption");
  const auto* wetValue = parameters.getRawParameterValue("fdtd_wet");
  const auto* dryValue = parameters.getRawParameterValue("fdtd_dry");

  const int sizeIndex = juce::jlimit(
      0, static_cast<int>(std::size(roomSizes)) - 1,
      static_cast<int>(
          std::round(sizeValue != nullptr ? sizeValue->load() : 1.0f)));

  int adjustedIndex = sizeIndex;
  if (lastBlockSize > 1024)
    adjustedIndex = 0;
  else if (lastBlockSize > 512)
    adjustedIndex = std::min(adjustedIndex, 1);

  const auto dims = roomSizes[adjustedIndex];

  ts::metal::fdtd::FDTDReverbConfig reverb{};
  reverb.solver.grid = {dims.nx, dims.ny, dims.nz};
  reverb.solver.dt = static_cast<float>(1.0 / sampleRate);
  reverb.solver.dx = 0.03f;
  reverb.solver.boundaryAttenuation =
      absorptionValue != nullptr ? absorptionValue->load() : 0.995f;
  reverb.solver.soundSpeed = 343.0f;
  reverb.solver.density = 1.2f;

  const int margin = 3;
  const int srcXRight = juce::jlimit(1, reverb.solver.grid.nx - margin,
                                     reverb.solver.grid.nx - margin);

  reverb.sources[0] = {
      static_cast<float>(juce::jlimit(1, reverb.solver.grid.nx - 2, margin)),
      static_cast<float>(margin), static_cast<float>(margin), 1.0f};
  reverb.sources[1] = {static_cast<float>(srcXRight),
                       static_cast<float>(margin), static_cast<float>(margin),
                       1.0f};

  const int centreX =
      juce::jlimit(1, reverb.solver.grid.nx - 2, reverb.solver.grid.nx / 2);
  const int centreY =
      juce::jlimit(1, reverb.solver.grid.ny - 2, reverb.solver.grid.ny / 2);
  const int centreZ =
      juce::jlimit(1, reverb.solver.grid.nz - 2, reverb.solver.grid.nz / 2);

  reverb.mics[0] = {static_cast<float>(centreX), static_cast<float>(centreY),
                    static_cast<float>(centreZ), 1.0f};
  reverb.mics[1] = {static_cast<float>(juce::jlimit(
                        1, reverb.solver.grid.nx - 2, centreX + 1)),
                    static_cast<float>(centreY), static_cast<float>(centreZ),
                    1.0f};

  reverb.wetLevel = wetValue != nullptr ? wetValue->load() : 1.0f;
  reverb.dryLevel = dryValue != nullptr ? dryValue->load() : 0.2f;

  std::string error;
  if (!fdtdEngine.prepare(reverb, error)) {
    const juce::SpinLock::ScopedLockType lock(fdtdStatusLock);
    fdtdStatus = "FDTD error: " + juce::String(error);
    fdtdEnabled.store(false);
    return;
  }

  fdtdEngine.setWatchdogTimeout(std::chrono::nanoseconds::zero());

  {
    const juce::SpinLock::ScopedLockType lock(fdtdStatusLock);
    fdtdStatus = juce::String(fdtdEngine.statusString());
  }
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
  return new PluginProcessor();
}
