#include "WaveguideCSVLoader.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>

namespace ts::metal::waveguide {

bool WaveguideCSVLoader::loadFromFile(const std::string& path,
                                      std::vector<WaveguideParams>& params,
                                      std::string& error,
                                      int maxDelayLength) {
    std::ifstream file(path);
    if (!file.is_open()) {
        error = "Failed to open file: " + path;
        return false;
    }

    params.clear();
    std::string line;
    int lineNumber = 0;

    while (std::getline(file, line)) {
        lineNumber++;

        if (line.empty() || line[0] == '#')
            continue;

        WaveguideParams p;
        std::string lineError;
        if (!parseLine(line, p, lineError, maxDelayLength)) {
            error = "Line " + std::to_string(lineNumber) + ": " + lineError;
            return false;
        }
        params.push_back(p);
    }

    if (params.empty()) {
        error = "No valid waveguide entries found";
        return false;
    }

    return true;
}

bool WaveguideCSVLoader::parseLine(const std::string& line,
                                   WaveguideParams& params,
                                   std::string& error,
                                   int maxDelayLength) {
    std::stringstream ss(line);
    std::string token;
    std::vector<float> values;

    while (std::getline(ss, token, ',')) {
        size_t start = token.find_first_not_of(" \t");
        size_t end = token.find_last_not_of(" \t");
        if (start == std::string::npos) {
            error = "Empty field";
            return false;
        }
        token = token.substr(start, end - start + 1);

        try {
            values.push_back(std::stof(token));
        } catch (...) {
            error = "Invalid number: " + token;
            return false;
        }
    }

    if (values.size() < 8) {
        error = "Expected 8 fields (id, length, inputTap, outputTap, filterCoeff, nonlinearity, gain, pan)";
        return false;
    }

    params.lengthSamples = static_cast<int>(values[1]);
    params.inputTap = std::clamp(values[2], 0.0f, 1.0f);
    params.outputTap = std::clamp(values[3], 0.0f, 1.0f);
    params.filterCoeff = std::clamp(values[4], 0.0f, 0.9999f);
    params.nonlinearityAmount = std::clamp(values[5], 0.0f, 1.0f);
    params.gain = values[6];
    params.pan = std::clamp(values[7], -1.0f, 1.0f);

    if (params.lengthSamples < 2) {
        error = "Length must be at least 2 samples";
        return false;
    }

    if (maxDelayLength > 0 && params.lengthSamples > maxDelayLength) {
        params.lengthSamples = maxDelayLength;
    }

    return true;
}

void WaveguideCSVLoader::generateRandom(std::vector<WaveguideParams>& params,
                                        int count,
                                        float sampleRate,
                                        int maxDelayLength,
                                        uint32_t seed) {
    if (count <= 0) return;

    params.clear();
    params.reserve(static_cast<size_t>(count));

    std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);
    std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);
    std::uniform_real_distribution<float> tapDist(0.05f, 0.95f);
    std::uniform_real_distribution<float> nonlinDist(0.1f, 0.5f);
    std::uniform_real_distribution<float> decayVariation(0.7f, 1.3f);

    const float minFreq = 40.0f;
    const float maxFreq = 1000.0f;
    const float baseDecaySeconds = 1.0f;

    float totalGain = 0.0f;
    std::vector<float> rawGains(static_cast<size_t>(count));

    for (int i = 0; i < count; i++) {
        WaveguideParams p;

        float freq = pinkNoiseFrequency(rng, minFreq, maxFreq);
        p.lengthSamples = static_cast<int>(sampleRate / freq);
        int maxLength = maxDelayLength > 0 ? maxDelayLength : 1200;
        p.lengthSamples = std::max(2, std::min(maxLength, p.lengthSamples));

        p.inputTap = tapDist(rng);
        p.outputTap = tapDist(rng);

        float decaySeconds = baseDecaySeconds * decayVariation(rng);
        p.filterCoeff = decayTimeToFilterCoeff(decaySeconds, p.lengthSamples, sampleRate);

        p.nonlinearityAmount = nonlinDist(rng);

        rawGains[static_cast<size_t>(i)] = uniform01(rng) + 0.5f;
        totalGain += rawGains[static_cast<size_t>(i)];

        float panBase = (count > 1)
            ? (static_cast<float>(i) / static_cast<float>(count - 1)) * 2.0f - 1.0f
            : 0.0f;
        p.pan = panBase * 0.8f + (uniform01(rng) - 0.5f) * 0.2f;
        p.pan = std::clamp(p.pan, -1.0f, 1.0f);

        params.push_back(p);
    }

    for (size_t i = 0; i < params.size(); i++) {
        params[i].gain = rawGains[i] / totalGain;
    }
}

float WaveguideCSVLoader::pinkNoiseFrequency(std::mt19937& rng,
                                             float minFreq,
                                             float maxFreq) {
    // 1/f distribution: freq = minFreq * (maxFreq/minFreq)^random
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    return minFreq * std::pow(maxFreq / minFreq, r);
}

float WaveguideCSVLoader::decayTimeToFilterCoeff(float decaySeconds,
                                                 int lengthSamples,
                                                 float sampleRate) {
    // RT60 decay: coefficient applied once per waveguide period
    // a^numPeriods = 0.001 (−60dB), solve for a
    float decaySamples = decaySeconds * sampleRate;
    float numPeriods = decaySamples / static_cast<float>(lengthSamples);
    float coeff = std::pow(0.001f, 1.0f / numPeriods);
    return std::clamp(coeff, 0.0f, 0.9999f);
}

}  // namespace ts::metal::waveguide
