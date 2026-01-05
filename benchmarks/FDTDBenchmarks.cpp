#include <ts_metal_accel_waveguide/ts_metal_accel_waveguide.h>

#include <array>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <string>

TEST_CASE("Waveguide Performance: 50 guides @ 512 samples", "[bench][waveguide]") {
    using namespace ts::metal::waveguide;

    WaveguideConfig config{};
    config.maxWaveguides = 50;
    config.sampleRate = 48000.0f;
    config.maxDelayLineSamples = 1200;

    WaveguideEngine engine;
    std::string error;
    REQUIRE(engine.prepare(config, error));

    engine.generateRandomPreset(42);

    constexpr int numSamples = 512;
    std::array<float, numSamples> inputL{};
    std::array<float, numSamples> inputR{};
    inputL[0] = 1.0f;
    inputR[0] = 0.5f;

    const float* inputs[2] = {inputL.data(), inputR.data()};
    std::array<float, numSamples> outputL{};
    std::array<float, numSamples> outputR{};
    float* outputs[2] = {outputL.data(), outputR.data()};

    BENCHMARK("Waveguide: 50 guides @ 512 samples") {
        engine.reset();
        engine.processBlock(inputs, 2, outputs, 2, numSamples);
    };
}
