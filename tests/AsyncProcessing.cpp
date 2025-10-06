#if defined(RUN_PAMPLEJUCE_TESTS)

#include <juce_audio_basics/juce_audio_basics.h>
#include <ts_metal_accel/ts_metal_accel.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <thread>

namespace {
constexpr auto kAsyncPassthroughKernel = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void async_passthrough(
    const device float* inBuffer [[buffer(0)]],
    device float* outBuffer [[buffer(1)]],
    constant uint& totalSamples [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= totalSamples)
        return;

    outBuffer[gid] = inBuffer[gid];
}
)METAL";
}  // namespace

TEST_CASE("MetalAccelerator async pipeline propagates audio", "[gpu][async]") {
  using namespace std::chrono_literals;

  ts::metal::MetalAccelerator accelerator;
  REQUIRE(accelerator.initialize());

  const std::string kernelName{"async_passthrough"};
  REQUIRE(accelerator.loadKernel(kernelName, kAsyncPassthroughKernel));

  ts::metal::MetalAccelerator::AsyncConfig config;
  config.maxChannels = 2;
  config.maxSamplesPerBlock = 32;
  config.queueDepth = 3;
  config.maxParameterBytes = 0;
  REQUIRE(accelerator.enableAsyncProcessing(config));

  juce::AudioBuffer<float> firstBlock(config.maxChannels,
                                      config.maxSamplesPerBlock);
  for (int ch = 0; ch < firstBlock.getNumChannels(); ++ch) {
    for (int sample = 0; sample < firstBlock.getNumSamples(); ++sample) {
      const float value =
          (ch == 0 ? 1.0f : -1.0f) * static_cast<float>(sample + 1);
      firstBlock.setSample(ch, sample, value);
    }
  }

  REQUIRE(accelerator.processBlockAsync(kernelName, firstBlock));

  juce::AudioBuffer<float> probeBlock(config.maxChannels,
                                      config.maxSamplesPerBlock);
  bool receivedExpectedOutput = false;

  for (int attempt = 0; attempt < 6 && !receivedExpectedOutput; ++attempt) {
    probeBlock.clear();

    auto result = accelerator.processBlockAsync(kernelName, probeBlock);
    REQUIRE(result);

    bool matches = true;
    for (int ch = 0; ch < probeBlock.getNumChannels(); ++ch) {
      for (int sample = 0; sample < probeBlock.getNumSamples(); ++sample) {
        const float expected = firstBlock.getSample(ch, sample);
        const float actual = probeBlock.getSample(ch, sample);

        if (actual != Catch::Approx(expected)) {
          matches = false;
          break;
        }
      }

      if (!matches) break;
    }

    receivedExpectedOutput = matches;

    if (!receivedExpectedOutput) std::this_thread::sleep_for(10ms);
  }

  accelerator.disableAsyncProcessing();

  REQUIRE(receivedExpectedOutput);
}

#endif  // RUN_PAMPLEJUCE_TESTS
