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
  PluginProcessor& processorRef;
  std::unique_ptr<melatonin::Inspector> inspector;
  juce::ToggleButton asyncToggle{"Enable Phase 2 (Async GPU)"};
  juce::ToggleButton waveguideToggle{"Enable Waveguide Reverb"};
  juce::Label statusLabel{"status", ""};
  juce::Label waveguideLabel{"waveguide", ""};
  juce::Label metricsLabel{"metrics", ""};

  void refreshStatus();
  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginEditor)
};
