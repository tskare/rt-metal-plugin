#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_events/juce_events.h>
#include <ts_metal_accel/ts_metal_accel.h>
#include <ts_metal_accel_fdtd/ts_metal_accel_fdtd.h>

#include <atomic>
#include <expected>
#include <string>

#if (MSVC)
#include "ipps.h"
#endif

class PluginProcessor : public juce::AudioProcessor,
                        private juce::AsyncUpdater,
                        private juce::AudioProcessorValueTreeState::Listener {
 public:
  PluginProcessor();
  ~PluginProcessor() override;

  void prepareToPlay(double sampleRate, int samplesPerBlock) override;
  void releaseResources() override;

  bool isBusesLayoutSupported(const BusesLayout& layouts) const override;

  void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

  juce::AudioProcessorEditor* createEditor() override;
  bool hasEditor() const override;

  const juce::String getName() const override;

  bool acceptsMidi() const override;
  bool producesMidi() const override;
  bool isMidiEffect() const override;
  double getTailLengthSeconds() const override;

  int getNumPrograms() override;
  int getCurrentProgram() override;
  void setCurrentProgram(int index) override;
  const juce::String getProgramName(int index) override;
  void changeProgramName(int index, const juce::String& newName) override;

  void getStateInformation(juce::MemoryBlock& destData) override;
  void setStateInformation(const void* data, int sizeInBytes) override;

  void setProcessingModeAsync(bool useAsync);
  bool isAsyncRequested() const;
  bool isAsyncActive() const;
  juce::String getProcessingStatus() const;
  ts::metal::MetalAccelerator::GPUStats getGPUStats() const;
  void setFDTDEnabled(bool enabled);
  bool isFDTDEnabled() const;
  juce::String getFDTDStatus() const;
  void setFDTDGPUPreferred(bool useGPU);
  bool isFDTDGPUPreferred() const;
  ts::metal::fdtd::FDTDReverbEngine::AsyncMetrics getFDTDAsyncMetrics() const;

  juce::AudioProcessorValueTreeState& getValueTree() { return parameters; }

  static juce::AudioProcessorValueTreeState::ParameterLayout
  createParameterLayout();

 private:
  struct DistortionParams {
    float drive = 4.0f;
    float mix = 0.75f;
    float outputGain = 0.5f;
  };

  void applyAsyncConfiguration();
  void updateStatus(juce::String newStatus);
  std::expected<void, ts::metal::MetalErrorInfo> processSynchronously(
      const std::string& kernelName, juce::AudioBuffer<float>& buffer,
      const void* params, size_t paramsSize);

  void handleAsyncUpdate() override;
  void requestAsyncTeardown();
  void parameterChanged(const juce::String& parameterID,
                        float newValue) override;
  void configureFDTD(double sampleRate);

  juce::AudioProcessorValueTreeState parameters;

  ts::metal::MetalAccelerator accelerator;
  std::string kernelName{"soft_clipper"};
  DistortionParams distortionParams;

  ts::metal::MetalAccelerator::AsyncConfig asyncConfig;
  std::atomic<bool> asyncRequested{false};
  std::atomic<bool> asyncActive{false};
  std::atomic<bool> asyncConfigPending{
      false};  // Signals config change from UI thread
  std::atomic<bool> asyncTeardownPending{false};

  double lastSampleRate = 44100.0;
  int lastBlockSize = 512;
  juce::String statusText{"Phase 1 (synchronous)"};
  mutable juce::SpinLock statusLock;

  std::atomic<bool> fdtdEnabled{false};
  ts::metal::fdtd::FDTDReverbEngine fdtdEngine;
  juce::String fdtdStatus{"FDTD disabled"};
  mutable juce::SpinLock fdtdStatusLock;
  std::atomic<bool> fdtdConfigDirty{false};
  std::atomic<bool> fdtdGPUPreferred{true};

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginProcessor)
};
