#pragma once

#include "BinaryData.h"
#include "PluginProcessor.h"
#include "melatonin_inspector/melatonin_inspector.h"

//==============================================================================
class PluginEditor : public juce::AudioProcessorEditor, private juce::Timer {
 public:
  explicit PluginEditor(PluginProcessor&);
  ~PluginEditor() override;

  //==============================================================================
  void paint(juce::Graphics&) override;
  void resized() override;
  void timerCallback() override;

 private:
  // This reference is provided as a quick way for your editor to
  // access the processor object that created it.
  PluginProcessor& processorRef;
  std::unique_ptr<melatonin::Inspector> inspector;
  juce::ToggleButton asyncToggle{"Enable Phase 2 (Async GPU)"};
  juce::ToggleButton fdtdToggle{"Enable FDTD Reverb"};
  juce::ToggleButton fdtdGpuToggle{"Process Reverb on GPU"};
  juce::ComboBox roomSizeBox;
  juce::Slider absorptionSlider;
  juce::Slider wetSlider;
  juce::Slider drySlider;
  juce::Label absorptionLabel{"absorptionLabel", "Absorption:"};
  juce::Label wetLabel{"wetLabel", "Wet:"};
  juce::Label dryLabel{"dryLabel", "Dry:"};
  juce::Label statusLabel{"status", ""};
  juce::Label metricsLabel{"metrics", ""};
  juce::Label fdtdLabel{"fdtd", ""};

  std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment>
      roomSizeAttachment;
  std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>
      absorptionAttachment;
  std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>
      wetAttachment;
  std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>
      dryAttachment;

  void refreshStatus();
  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginEditor)
};
