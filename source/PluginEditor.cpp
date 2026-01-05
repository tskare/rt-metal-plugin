#include "PluginEditor.h"

PluginEditor::PluginEditor(PluginProcessor& p)
    : AudioProcessorEditor(&p), processorRef(p) {
  juce::ignoreUnused(processorRef);

  addAndMakeVisible(asyncToggle);
  addAndMakeVisible(waveguideToggle);
  addAndMakeVisible(statusLabel);
  addAndMakeVisible(waveguideLabel);
  addAndMakeVisible(metricsLabel);

  asyncToggle.setColour(juce::ToggleButton::textColourId,
                        juce::Colour(0xffe0e0e0));
  waveguideToggle.setColour(juce::ToggleButton::textColourId,
                            juce::Colour(0xffe0e0e0));

  asyncToggle.setToggleState(processorRef.isAsyncRequested(),
                             juce::dontSendNotification);
  asyncToggle.onClick = [this]() {
    processorRef.setProcessingModeAsync(asyncToggle.getToggleState());
    refreshStatus();
  };

  waveguideToggle.setToggleState(processorRef.isWaveguideEnabled(),
                                 juce::dontSendNotification);
  waveguideToggle.onClick = [this]() {
    processorRef.setWaveguideEnabled(waveguideToggle.getToggleState());
    refreshStatus();
  };

  statusLabel.setJustificationType(juce::Justification::centred);
  statusLabel.setFont(juce::FontOptions(
      juce::Font::getDefaultSansSerifFontName(), 14.0f, juce::Font::plain));
  statusLabel.setColour(juce::Label::textColourId, juce::Colour(0xffe0e0e0));

  waveguideLabel.setJustificationType(juce::Justification::centred);
  waveguideLabel.setFont(juce::FontOptions(
      juce::Font::getDefaultSansSerifFontName(), 14.0f, juce::Font::plain));
  waveguideLabel.setColour(juce::Label::textColourId, juce::Colour(0xffe0e0e0));

  metricsLabel.setJustificationType(juce::Justification::centredLeft);
  metricsLabel.setFont(juce::FontOptions(
      juce::Font::getDefaultMonospacedFontName(), 13.0f, juce::Font::plain));
  metricsLabel.setColour(juce::Label::textColourId, juce::Colour(0xffe0e0e0));

  refreshStatus();
  startTimerHz(10);

  setSize(500, 450);
  setResizable(true, true);
  setResizeLimits(400, 350, 800, 700);
}

PluginEditor::~PluginEditor() { stopTimer(); }

void PluginEditor::paint(juce::Graphics& g) {
  g.fillAll(juce::Colour(0xff1e1e1e));
}

void PluginEditor::resized() {
  auto area = getLocalBounds().reduced(20);

  auto toggleArea = area.removeFromTop(40);
  asyncToggle.setBounds(toggleArea);

  auto waveguideToggleArea = area.removeFromTop(40);
  waveguideToggle.setBounds(waveguideToggleArea);

  auto statusArea = area.removeFromTop(30);
  statusLabel.setBounds(statusArea);

  auto waveguideStatusArea = area.removeFromTop(30);
  waveguideLabel.setBounds(waveguideStatusArea);

  auto metricsArea = area.removeFromTop(250);
  metricsLabel.setBounds(metricsArea);
}

void PluginEditor::timerCallback() { refreshStatus(); }

void PluginEditor::refreshStatus() {
  statusLabel.setText(processorRef.getProcessingStatus(),
                      juce::dontSendNotification);
  asyncToggle.setToggleState(processorRef.isAsyncRequested(),
                             juce::dontSendNotification);
  waveguideToggle.setToggleState(processorRef.isWaveguideEnabled(),
                                 juce::dontSendNotification);
  waveguideLabel.setText(processorRef.getWaveguideStatus(),
                         juce::dontSendNotification);

  juce::String metricsText;

  auto stats = processorRef.getGPUStats();
  if (stats.asyncEnabled) {
    metricsText << "GPU Metrics (Soft Clipper)\n";
    if (stats.warmupFramesRemaining > 0)
      metricsText << "  Warmup: " << stats.warmupFramesRemaining
                  << " frame(s) remaining\n";
    metricsText << "  Frames: " << stats.totalFrames << "\n";
    metricsText << "  Underruns: " << stats.underrunCount << " ("
                << juce::String(stats.underrunRate * 100.0, 2) << "%)\n";
  } else {
    metricsText << "GPU Metrics (Soft Clipper)\n  N/A (sync mode)\n";
  }

  metricsText << "\nWaveguide Engine\n";
  if (processorRef.isWaveguideEnabled()) {
    metricsText << "  Waveguides: " << processorRef.getWaveguideCount() << "\n";
    metricsText << "  Status: Active\n";
  } else {
    metricsText << "  Status: Disabled (using soft clipper fallback)\n";
  }

  metricsLabel.setText(metricsText.trimEnd(), juce::dontSendNotification);
}
