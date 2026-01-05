#pragma once

#include "WaveguideTypes.h"
#include "WaveguideCSVLoader.h"
#include <string>
#include <vector>
#include <memory>

namespace ts::metal::waveguide {

class WaveguideEngine {
public:
    WaveguideEngine();
    ~WaveguideEngine();

    // Non-copyable
    WaveguideEngine(const WaveguideEngine&) = delete;
    WaveguideEngine& operator=(const WaveguideEngine&) = delete;

    bool prepare(const WaveguideConfig& config, std::string& error);

    bool loadPreset(const std::string& csvPath, std::string& error);

    void generateRandomPreset(uint32_t seed = 0);

    void processBlock(const float* const* inputs,
                      int numInputChannels,
                      float* const* outputs,
                      int numOutputChannels,
                      int numSamples);

    void reset();
    void shutdown();

    int getWaveguideCount() const { return static_cast<int>(params_.size()); }
    bool isInitialized() const { return initialized_; }
    const char* statusString() const;

private:
    bool initializeMetal(std::string& error);
    bool allocateBuffers(std::string& error);
    void uploadParams();
    void processChunk(const float* inL, const float* inR,
                      float* outL, float* outR, int numSamples);

    WaveguideConfig config_;
    std::vector<WaveguideParams> params_;
    bool initialized_ = false;
    std::string status_;

    // Metal resources (forward declared, implemented in .cpp)
    struct MetalState;
    std::unique_ptr<MetalState> metal_;
};

}  // namespace ts::metal::waveguide
