#pragma once

#include "WaveguideTypes.h"
#include <string>
#include <vector>
#include <random>

namespace ts::metal::waveguide {

class WaveguideCSVLoader {
public:
    static bool loadFromFile(const std::string& path,
                             std::vector<WaveguideParams>& params,
                             std::string& error,
                             int maxDelayLength = 0);

    static void generateRandom(std::vector<WaveguideParams>& params,
                               int count,
                               float sampleRate,
                               int maxDelayLength,
                               uint32_t seed = 0);

private:
    static bool parseLine(const std::string& line,
                          WaveguideParams& params,
                          std::string& error,
                          int maxDelayLength);

    static float pinkNoiseFrequency(std::mt19937& rng,
                                    float minFreq,
                                    float maxFreq);

    static float decayTimeToFilterCoeff(float decaySeconds,
                                        int lengthSamples,
                                        float sampleRate);
};

}  // namespace ts::metal::waveguide
