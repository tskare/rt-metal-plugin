#include "PluginEditor.h"

PluginEditor::PluginEditor(PluginProcessor& p)
    : AudioProcessorEditor(&p), processorRef(p) {
  juce::ignoreUnused(processorRef);

  addAndMakeVisible(asyncToggle);
  addAndMakeVisible(fdtdToggle);
  addAndMakeVisible(fdtdGpuToggle);
  addAndMakeVisible(roomSizeBox);
  addAndMakeVisible(absorptionSlider);
  addAndMakeVisible(wetSlider);
  addAndMakeVisible(drySlider);
  addAndMakeVisible(absorptionLabel);
  addAndMakeVisible(wetLabel);
  addAndMakeVisible(dryLabel);
  addAndMakeVisible(statusLabel);
  addAndMakeVisible(metricsLabel);
  addAndMakeVisible(fdtdLabel);

  // Set dark mode colors for toggle buttons
  asyncToggle.setColour(juce::ToggleButton::textColourId,
                        juce::Colour(0xffe0e0e0));
  fdtdToggle.setColour(juce::ToggleButton::textColourId,
                       juce::Colour(0xffe0e0e0));
  fdtdGpuToggle.setColour(juce::ToggleButton::textColourId,
                          juce::Colour(0xffe0e0e0));

  asyncToggle.setToggleState(processorRef.isAsyncRequested(),
                             juce::dontSendNotification);
  asyncToggle.onClick = [this]() {
    processorRef.setProcessingModeAsync(asyncToggle.getToggleState());
    refreshStatus();
  };

  fdtdToggle.setToggleState(processorRef.isFDTDEnabled(),
                            juce::dontSendNotification);
  fdtdToggle.onClick = [this]() {
    processorRef.setFDTDEnabled(fdtdToggle.getToggleState());
    refreshStatus();
  };

  fdtdGpuToggle.setToggleState(processorRef.isFDTDGPUPreferred(),
                               juce::dontSendNotification);
  fdtdGpuToggle.onClick = [this]() {
    processorRef.setFDTDGPUPreferred(fdtdGpuToggle.getToggleState());
    refreshStatus();
  };

  // Use San Francisco (system font) for better readability with slightly larger
  // sizes
  statusLabel.setJustificationType(juce::Justification::centred);
  statusLabel.setFont(juce::FontOptions(
      juce::Font::getDefaultSansSerifFontName(), 14.0f, juce::Font::plain));
  statusLabel.setColour(juce::Label::textColourId, juce::Colour(0xffe0e0e0));

  metricsLabel.setJustificationType(juce::Justification::centredLeft);
  metricsLabel.setFont(juce::FontOptions(
      juce::Font::getDefaultMonospacedFontName(), 13.0f, juce::Font::plain));
  metricsLabel.setColour(juce::Label::textColourId, juce::Colour(0xffe0e0e0));

  fdtdLabel.setJustificationType(juce::Justification::centred);
  fdtdLabel.setFont(juce::FontOptions(juce::Font::getDefaultSansSerifFontName(),
                                      14.0f, juce::Font::plain));
  fdtdLabel.setColour(juce::Label::textColourId, juce::Colour(0xffe0e0e0));

  // Style slider labels
  auto configureSliderLabel = [](juce::Label& label) {
    label.setJustificationType(juce::Justification::centredRight);
    label.setFont(juce::FontOptions(juce::Font::getDefaultSansSerifFontName(),
                                    14.0f, juce::Font::plain));
    label.setColour(juce::Label::textColourId, juce::Colour(0xffe0e0e0));
  };
  configureSliderLabel(absorptionLabel);
  configureSliderLabel(wetLabel);
  configureSliderLabel(dryLabel);

  roomSizeBox.addItemList(juce::StringArray{"Small", "Medium", "Large"}, 1);
  roomSizeBox.setJustificationType(juce::Justification::centred);
  roomSizeBox.setColour(juce::ComboBox::textColourId, juce::Colour(0xffe0e0e0));
  roomSizeBox.setColour(juce::ComboBox::backgroundColourId,
                        juce::Colour(0xff2d2d2d));
  roomSizeBox.setColour(juce::ComboBox::arrowColourId,
                        juce::Colour(0xffe0e0e0));

  auto configureSlider = [](juce::Slider& slider, float defaultValue) {
    slider.setSliderStyle(juce::Slider::LinearHorizontal);
    slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 80, 20);
    slider.setDoubleClickReturnValue(true, defaultValue);
    // Dark mode colors for sliders
    slider.setColour(juce::Slider::textBoxTextColourId,
                     juce::Colour(0xffe0e0e0));
    slider.setColour(juce::Slider::textBoxBackgroundColourId,
                     juce::Colour(0xff2d2d2d));
    slider.setColour(juce::Slider::textBoxOutlineColourId,
                     juce::Colour(0xff3d3d3d));
    slider.setColour(juce::Slider::trackColourId, juce::Colour(0xff007acc));
    slider.setColour(juce::Slider::backgroundColourId,
                     juce::Colour(0xff3d3d3d));
    slider.setColour(juce::Slider::thumbColourId, juce::Colour(0xff007acc));
  };

  configureSlider(absorptionSlider, 0.995f);
  configureSlider(wetSlider, 1.0f);
  configureSlider(drySlider, 0.2f);

  absorptionSlider.setTooltip(
      "Boundary absorption (closer to 1.0 = longer decay)");
  wetSlider.setTooltip("Wet gain");
  drySlider.setTooltip("Dry gain");

  auto& valueTree = processorRef.getValueTree();
  roomSizeAttachment =
      std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
          valueTree, "room_size", roomSizeBox);
  absorptionAttachment =
      std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
          valueTree, "fdtd_absorption", absorptionSlider);
  wetAttachment =
      std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
          valueTree, "fdtd_wet", wetSlider);
  dryAttachment =
      std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
          valueTree, "fdtd_dry", drySlider);

  refreshStatus();
  startTimerHz(10);

  // Make sure that before the constructor has finished, you've set the
  // editor's size to whatever you need it to be.
  setSize(600, 650);
  setResizable(true, true);
  setResizeLimits(500, 500, 1200, 1200);
}

PluginEditor::~PluginEditor() { stopTimer(); }

void PluginEditor::paint(juce::Graphics& g) {
  // Professional dark mode background
  g.fillAll(juce::Colour(0xff1e1e1e));
}

void PluginEditor::resized() {
  auto area = getLocalBounds().reduced(20);

  auto toggleArea = area.removeFromTop(40);
  asyncToggle.setBounds(toggleArea.removeFromLeft(toggleArea.getWidth()));

  auto fdtdToggleArea = area.removeFromTop(40);
  fdtdToggle.setBounds(
      fdtdToggleArea.removeFromLeft(fdtdToggleArea.getWidth()));

  auto fdtdGpuToggleArea = area.removeFromTop(30);
  fdtdGpuToggle.setBounds(
      fdtdGpuToggleArea.removeFromLeft(fdtdGpuToggleArea.getWidth()));

  auto parameterArea = area.removeFromTop(120);
  auto comboArea = parameterArea.removeFromTop(30);
  roomSizeBox.setBounds(comboArea.removeFromLeft(comboArea.getWidth()));

  auto absorptionArea = parameterArea.removeFromTop(30);
  absorptionLabel.setBounds(absorptionArea.removeFromLeft(100));
  absorptionSlider.setBounds(absorptionArea);

  auto wetArea = parameterArea.removeFromTop(30);
  wetLabel.setBounds(wetArea.removeFromLeft(100));
  wetSlider.setBounds(wetArea);

  auto dryArea = parameterArea.removeFromTop(30);
  dryLabel.setBounds(dryArea.removeFromLeft(100));
  drySlider.setBounds(dryArea);

  auto statusArea = area.removeFromTop(40);
  statusLabel.setBounds(statusArea);

  auto fdtdStatusArea = area.removeFromTop(40);
  fdtdLabel.setBounds(fdtdStatusArea);

  auto metricsArea = area.removeFromTop(300);
  metricsLabel.setBounds(metricsArea);
}

void PluginEditor::timerCallback() { refreshStatus(); }

void PluginEditor::refreshStatus() {
  statusLabel.setText(processorRef.getProcessingStatus(),
                      juce::dontSendNotification);
  asyncToggle.setToggleState(processorRef.isAsyncRequested(),
                             juce::dontSendNotification);
  fdtdToggle.setToggleState(processorRef.isFDTDEnabled(),
                            juce::dontSendNotification);
  fdtdLabel.setText(processorRef.getFDTDStatus(), juce::dontSendNotification);
  fdtdGpuToggle.setToggleState(processorRef.isFDTDGPUPreferred(),
                               juce::dontSendNotification);
  fdtdGpuToggle.setEnabled(processorRef.isFDTDEnabled());

  // Update GPU metrics display
  juce::String metricsText;

  auto stats = processorRef.getGPUStats();
  if (stats.asyncEnabled) {
    metricsText << "GPU Metrics\n";
    if (stats.warmupFramesRemaining > 0)
      metricsText << "  Warmup: " << stats.warmupFramesRemaining
                  << " frame(s) remaining\n";
    metricsText << "  Frames: " << stats.totalFrames << "\n";
    metricsText << "  Underruns: " << stats.underrunCount << " ("
                << juce::String(stats.underrunRate * 100.0, 2) << "%)\n";
  } else {
    metricsText << "GPU Metrics\n  N/A (sync mode)\n";
  }

  metricsText << "\n";

  auto fdtdMetrics = processorRef.getFDTDAsyncMetrics();
  if (fdtdMetrics.asyncEnabled) {
    metricsText << "FDTD GPU Queue\n";
    if (fdtdMetrics.warmupFramesRemaining > 0)
      metricsText << "  Warmup: " << fdtdMetrics.warmupFramesRemaining
                  << " frame(s)\n";
    metricsText << "  Depth: " << static_cast<int>(fdtdMetrics.queueDepth)
                << "\n";
    metricsText << "  Frames: " << fdtdMetrics.completedFrames << "\n";
    metricsText << "  processBlock(): " << fdtdMetrics.processBlockCalls
                << "\n";
    metricsText << "  Enqueue attempts: " << fdtdMetrics.enqueueAttempts
                << "\n";
    metricsText << "  Enqueue drops: " << fdtdMetrics.enqueueDrops << "\n";
    metricsText << "  Underruns: " << fdtdMetrics.underruns << "\n";
    metricsText << "  Watchdogs: " << fdtdMetrics.watchdogTrips << "\n";
    metricsText << "  Latency samples: " << fdtdMetrics.latencySampleCount
                << "\n";

    if (fdtdMetrics.warmupFramesRemaining == 0 &&
        fdtdMetrics.latencySampleCount >= 10) {
      metricsText << "  Avg ms: "
                  << juce::String(fdtdMetrics.averageLatencyMillis, 3) << "\n";
      metricsText << "  Max ms: "
                  << juce::String(fdtdMetrics.maxLatencyMillis, 3) << "\n";
      metricsText << "  CPU enqueue ms: "
                  << juce::String(fdtdMetrics.cpuEnqueueMillis, 3) << "\n";
      metricsText << "  CPU copy ms: "
                  << juce::String(fdtdMetrics.cpuCopyMillis, 3) << "\n";
    } else {
      metricsText << "  Avg ms: warming up...\n";
      metricsText << "  Max ms: warming up...\n";
      metricsText << "  CPU enqueue ms: warming up...\n";
      metricsText << "  CPU copy ms: warming up...\n";
    }

    if (fdtdMetrics.watchdogTimeoutMillis > 0.0)
      metricsText << "  Watchdog ms: "
                  << juce::String(fdtdMetrics.watchdogTimeoutMillis, 3) << "\n";
    else
      metricsText << "  Watchdog: disabled\n";
  } else {
    metricsText << "FDTD GPU Queue\n  Idle (CPU path active)\n";
  }

  metricsLabel.setText(metricsText.trimEnd(), juce::dontSendNotification);
}
