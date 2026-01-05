#include <ts_metal_accel_waveguide/ts_metal_accel_waveguide.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <fstream>
#include <cmath>

using namespace ts::metal::waveguide;

TEST_CASE("WaveguideCSVLoader parses valid CSV", "[waveguide][csv]") {
    const char* testCSV = R"(# Test waveguide preset
0,100,0.1,0.9,0.8,0.3,0.5,-0.5
1,200,0.2,0.8,0.7,0.25,0.3,0.5
)";
    std::string tempPath = "/tmp/test_waveguide.csv";
    {
        std::ofstream f(tempPath);
        f << testCSV;
    }

    std::vector<WaveguideParams> params;
    std::string error;
    REQUIRE(WaveguideCSVLoader::loadFromFile(tempPath, params, error));
    REQUIRE(params.size() == 2);

    CHECK(params[0].lengthSamples == 100);
    CHECK(params[0].inputTap == Catch::Approx(0.1f));
    CHECK(params[0].outputTap == Catch::Approx(0.9f));
    CHECK(params[0].filterCoeff == Catch::Approx(0.8f));
    CHECK(params[0].nonlinearityAmount == Catch::Approx(0.3f));
    CHECK(params[0].gain == Catch::Approx(0.5f));
    CHECK(params[0].pan == Catch::Approx(-0.5f));

    CHECK(params[1].lengthSamples == 200);
    CHECK(params[1].pan == Catch::Approx(0.5f));

    std::remove(tempPath.c_str());
}

TEST_CASE("WaveguideCSVLoader clamps values", "[waveguide][csv]") {
    const char* testCSV = R"(0,100,1.5,1.5,1.5,1.5,1.0,2.0
)";
    std::string tempPath = "/tmp/test_waveguide_clamp.csv";
    {
        std::ofstream f(tempPath);
        f << testCSV;
    }

    std::vector<WaveguideParams> params;
    std::string error;
    REQUIRE(WaveguideCSVLoader::loadFromFile(tempPath, params, error));
    REQUIRE(params.size() == 1);

    CHECK(params[0].inputTap == Catch::Approx(1.0f));
    CHECK(params[0].outputTap == Catch::Approx(1.0f));
    CHECK(params[0].filterCoeff == Catch::Approx(0.9999f));
    CHECK(params[0].nonlinearityAmount == Catch::Approx(1.0f));
    CHECK(params[0].pan == Catch::Approx(1.0f));

    std::remove(tempPath.c_str());
}

TEST_CASE("WaveguideCSVLoader enforces maxDelayLength", "[waveguide][csv]") {
    const char* testCSV = R"(0,5000,0.5,0.5,0.5,0.5,1.0,0.0
)";
    std::string tempPath = "/tmp/test_waveguide_maxdelay.csv";
    {
        std::ofstream f(tempPath);
        f << testCSV;
    }

    std::vector<WaveguideParams> params;
    std::string error;
    REQUIRE(WaveguideCSVLoader::loadFromFile(tempPath, params, error, 1200));
    REQUIRE(params.size() == 1);

    CHECK(params[0].lengthSamples == 1200);

    std::remove(tempPath.c_str());
}

TEST_CASE("WaveguideCSVLoader rejects invalid files", "[waveguide][csv]") {
    std::vector<WaveguideParams> params;
    std::string error;

    SECTION("missing file") {
        REQUIRE_FALSE(WaveguideCSVLoader::loadFromFile("/nonexistent.csv", params, error));
    }

    SECTION("too few fields") {
        const char* csv = "0,100,0.5\n";
        std::string tempPath = "/tmp/test_waveguide_invalid.csv";
        {
            std::ofstream f(tempPath);
            f << csv;
        }
        REQUIRE_FALSE(WaveguideCSVLoader::loadFromFile(tempPath, params, error));
        std::remove(tempPath.c_str());
    }

    SECTION("invalid number") {
        const char* csv = "0,abc,0.5,0.5,0.5,0.5,1.0,0.0\n";
        std::string tempPath = "/tmp/test_waveguide_nan.csv";
        {
            std::ofstream f(tempPath);
            f << csv;
        }
        REQUIRE_FALSE(WaveguideCSVLoader::loadFromFile(tempPath, params, error));
        std::remove(tempPath.c_str());
    }

    SECTION("length too short") {
        const char* csv = "0,1,0.5,0.5,0.5,0.5,1.0,0.0\n";
        std::string tempPath = "/tmp/test_waveguide_short.csv";
        {
            std::ofstream f(tempPath);
            f << csv;
        }
        REQUIRE_FALSE(WaveguideCSVLoader::loadFromFile(tempPath, params, error));
        std::remove(tempPath.c_str());
    }
}

TEST_CASE("WaveguideCSVLoader generates random presets", "[waveguide][random]") {
    std::vector<WaveguideParams> params;

    SECTION("deterministic with seed") {
        WaveguideCSVLoader::generateRandom(params, 10, 48000.f, 1200, 12345);

        std::vector<WaveguideParams> params2;
        WaveguideCSVLoader::generateRandom(params2, 10, 48000.f, 1200, 12345);

        REQUIRE(params.size() == 10);
        REQUIRE(params2.size() == 10);

        for (size_t i = 0; i < params.size(); ++i) {
            CHECK(params[i].lengthSamples == params2[i].lengthSamples);
            CHECK(params[i].filterCoeff == params2[i].filterCoeff);
            CHECK(params[i].gain == params2[i].gain);
        }
    }

    SECTION("frequency range 40-1000Hz") {
        WaveguideCSVLoader::generateRandom(params, 50, 48000.f, 1200, 42);

        for (const auto& p : params) {
            float freq = 48000.f / static_cast<float>(p.lengthSamples);
            CHECK(freq >= 40.f);
            CHECK(freq <= 24000.f);
            CHECK(p.lengthSamples >= 2);
            CHECK(p.lengthSamples <= 1200);
        }
    }

    SECTION("gains sum to approximately 1") {
        WaveguideCSVLoader::generateRandom(params, 50, 48000.f, 1200, 99);

        float totalGain = 0.f;
        for (const auto& p : params) {
            totalGain += p.gain;
        }
        CHECK(totalGain == Catch::Approx(1.0f).margin(0.01f));
    }

    SECTION("pans distributed across stereo field") {
        WaveguideCSVLoader::generateRandom(params, 50, 48000.f, 1200, 77);

        int leftCount = 0, rightCount = 0;
        for (const auto& p : params) {
            CHECK(p.pan >= -1.f);
            CHECK(p.pan <= 1.f);
            if (p.pan < 0) leftCount++;
            else rightCount++;
        }
        CHECK(leftCount > 10);
        CHECK(rightCount > 10);
    }
}

#if defined(RUN_PAMPLEJUCE_TESTS)

#include <ts_metal_accel/ts_metal_accel.h>

TEST_CASE("WaveguideEngine initializes with Metal", "[waveguide][gpu]") {
    if (!ts::metal::isMetalAvailable()) {
        WARN("Metal unavailable; skipping GPU waveguide test");
        return;
    }

    WaveguideEngine engine;
    WaveguideConfig config;
    config.maxWaveguides = 10;
    config.sampleRate = 48000.f;
    config.maxDelayLineSamples = 1200;

    std::string error;
    REQUIRE(engine.prepare(config, error));
    CHECK(engine.isInitialized());
    CHECK(engine.getWaveguideCount() == 10);
}

TEST_CASE("WaveguideEngine processes audio", "[waveguide][gpu]") {
    if (!ts::metal::isMetalAvailable()) {
        WARN("Metal unavailable; skipping GPU waveguide test");
        return;
    }

    WaveguideEngine engine;
    WaveguideConfig config;
    config.maxWaveguides = 10;
    config.sampleRate = 48000.f;
    config.maxDelayLineSamples = 1200;

    std::string error;
    REQUIRE(engine.prepare(config, error));

    constexpr int numSamples = 512;
    std::array<float, numSamples> inL{}, inR{}, outL{}, outR{};
    inL[0] = 1.0f;
    inR[0] = 1.0f;

    const float* inputs[2] = {inL.data(), inR.data()};
    float* outputs[2] = {outL.data(), outR.data()};

    engine.processBlock(inputs, 2, outputs, 2, numSamples);

    bool hasOutput = false;
    for (int i = 0; i < numSamples; ++i) {
        if (outL[i] != 0.f || outR[i] != 0.f) {
            hasOutput = true;
            break;
        }
    }
    CHECK(hasOutput);
}

TEST_CASE("WaveguideEngine handles large blocks via chunking", "[waveguide][gpu]") {
    if (!ts::metal::isMetalAvailable()) {
        WARN("Metal unavailable; skipping GPU waveguide test");
        return;
    }

    WaveguideEngine engine;
    WaveguideConfig config;
    config.maxWaveguides = 10;
    config.sampleRate = 48000.f;
    config.maxDelayLineSamples = 1200;

    std::string error;
    REQUIRE(engine.prepare(config, error));

    constexpr int numSamples = 4096;
    std::vector<float> inL(numSamples, 0.f), inR(numSamples, 0.f);
    std::vector<float> outL(numSamples, 0.f), outR(numSamples, 0.f);
    inL[0] = 1.0f;
    inR[0] = 1.0f;

    const float* inputs[2] = {inL.data(), inR.data()};
    float* outputs[2] = {outL.data(), outR.data()};

    engine.processBlock(inputs, 2, outputs, 2, numSamples);

    bool hasOutputLateInBuffer = false;
    for (int i = 2048; i < numSamples; ++i) {
        if (outL[i] != 0.f || outR[i] != 0.f) {
            hasOutputLateInBuffer = true;
            break;
        }
    }
    CHECK(hasOutputLateInBuffer);
}

TEST_CASE("WaveguideEngine produces bounded output", "[waveguide][gpu]") {
    if (!ts::metal::isMetalAvailable()) {
        WARN("Metal unavailable; skipping GPU waveguide test");
        return;
    }

    WaveguideEngine engine;
    WaveguideConfig config;
    config.maxWaveguides = 50;
    config.sampleRate = 48000.f;
    config.maxDelayLineSamples = 1200;

    std::string error;
    REQUIRE(engine.prepare(config, error));

    constexpr int numBlocks = 100;
    constexpr int blockSize = 512;
    std::array<float, blockSize> inL{}, inR{}, outL{}, outR{};

    const float* inputs[2] = {inL.data(), inR.data()};
    float* outputs[2] = {outL.data(), outR.data()};

    float maxOutput = 0.f;
    for (int block = 0; block < numBlocks; ++block) {
        std::fill(inL.begin(), inL.end(), 0.f);
        std::fill(inR.begin(), inR.end(), 0.f);
        if (block == 0) {
            inL[0] = 1.0f;
            inR[0] = 1.0f;
        }

        engine.processBlock(inputs, 2, outputs, 2, blockSize);

        for (int i = 0; i < blockSize; ++i) {
            maxOutput = std::max(maxOutput, std::abs(outL[i]));
            maxOutput = std::max(maxOutput, std::abs(outR[i]));
        }
    }

    CHECK(maxOutput > 0.f);
    CHECK(maxOutput < 10.f);
}

#endif  // RUN_PAMPLEJUCE_TESTS
