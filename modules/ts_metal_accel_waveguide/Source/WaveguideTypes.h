#pragma once

#include <cstdint>

namespace ts::metal::waveguide {

struct WaveguideConfig {
    int maxWaveguides = 50;
    float sampleRate = 48000.f;
    int maxDelayLineSamples = 1200;  // longest delay line (48kHz / 40Hz)
};

struct WaveguideParams {
    int lengthSamples;           // delay line length in samples
    float inputTap;              // 0.0-1.0 position for input injection
    float outputTap;             // 0.0-1.0 position for output reading
    float filterCoeff;           // one-pole lowpass coefficient (0.0-1.0)
    float nonlinearityAmount;    // saturation strength (0.0 = off, 1.0 = full)
    float gain;                  // output gain
    float pan;                   // stereo pan (-1.0 L to +1.0 R)
};

struct WaveguideUniforms {
    uint32_t numWaveguides;
    uint32_t numSamples;
    uint32_t maxDelayLength;
    uint32_t halfWaveguideCount;  // waveguides 0..half-1 get L, half..end get R
};

}  // namespace ts::metal::waveguide
