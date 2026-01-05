#include "WaveguideEngine.h"
#include "WaveguideMetalKernels.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <cstring>
#include <algorithm>

namespace ts::metal::waveguide {

struct WaveguideEngine::MetalState {
    MTL::Device* device = nullptr;
    MTL::CommandQueue* commandQueue = nullptr;
    MTL::ComputePipelineState* processKernel = nullptr;
    MTL::ComputePipelineState* clearKernel = nullptr;

    MTL::Buffer* delayLines = nullptr;
    MTL::Buffer* paramsBuffer = nullptr;
    MTL::Buffer* filterState = nullptr;
    MTL::Buffer* writePositions = nullptr;
    MTL::Buffer* inputL = nullptr;
    MTL::Buffer* inputR = nullptr;
    MTL::Buffer* outputL = nullptr;
    MTL::Buffer* outputR = nullptr;

    int maxSamplesPerBlock = 0;

    ~MetalState() {
        if (delayLines) delayLines->release();
        if (paramsBuffer) paramsBuffer->release();
        if (filterState) filterState->release();
        if (writePositions) writePositions->release();
        if (inputL) inputL->release();
        if (inputR) inputR->release();
        if (outputL) outputL->release();
        if (outputR) outputR->release();
        if (processKernel) processKernel->release();
        if (clearKernel) clearKernel->release();
        if (commandQueue) commandQueue->release();
        if (device) device->release();
    }
};

WaveguideEngine::WaveguideEngine() : metal_(std::make_unique<MetalState>()) {}

WaveguideEngine::~WaveguideEngine() {
    shutdown();
}

bool WaveguideEngine::prepare(const WaveguideConfig& config, std::string& error) {
    config_ = config;

    if (!initializeMetal(error)) {
        return false;
    }

    if (params_.empty()) {
        generateRandomPreset();
    }

    if (!allocateBuffers(error)) {
        return false;
    }
    uploadParams();
    reset();

    initialized_ = true;
    status_ = "Waveguide GPU ready (" + std::to_string(params_.size()) + " waveguides)";
    return true;
}

bool WaveguideEngine::initializeMetal(std::string& error) {
    metal_->device = MTL::CreateSystemDefaultDevice();
    if (!metal_->device) {
        error = "Failed to create Metal device";
        return false;
    }

    metal_->commandQueue = metal_->device->newCommandQueue();
    if (!metal_->commandQueue) {
        error = "Failed to create command queue";
        return false;
    }

    NS::Error* compileError = nullptr;
    auto source = NS::String::string(kWaveguideKernelSource, NS::UTF8StringEncoding);
    auto library = metal_->device->newLibrary(source, nullptr, &compileError);

    if (!library) {
        error = "Failed to compile Metal shaders";
        if (compileError) {
            auto desc = compileError->localizedDescription();
            if (desc) {
                error += ": ";
                error += desc->utf8String();
            }
            compileError->release();
        }
        return false;
    }

    auto processFunc = library->newFunction(NS::String::string("waveguide_process_block", NS::UTF8StringEncoding));
    auto clearFunc = library->newFunction(NS::String::string("waveguide_clear_output", NS::UTF8StringEncoding));

    if (!processFunc || !clearFunc) {
        error = "Failed to find kernel functions";
        if (processFunc) processFunc->release();
        if (clearFunc) clearFunc->release();
        library->release();
        return false;
    }

    NS::Error* pipelineError = nullptr;
    metal_->processKernel = metal_->device->newComputePipelineState(processFunc, &pipelineError);
    if (pipelineError) pipelineError->release();

    pipelineError = nullptr;
    metal_->clearKernel = metal_->device->newComputePipelineState(clearFunc, &pipelineError);
    if (pipelineError) pipelineError->release();

    processFunc->release();
    clearFunc->release();
    library->release();

    if (!metal_->processKernel || !metal_->clearKernel) {
        error = "Failed to create pipeline states";
        return false;
    }

    return true;
}

bool WaveguideEngine::allocateBuffers(std::string& error) {
    const size_t numWaveguides = params_.size();
    const size_t maxDelay = static_cast<size_t>(config_.maxDelayLineSamples);
    const size_t maxSamples = 2048;

    metal_->maxSamplesPerBlock = static_cast<int>(maxSamples);

    size_t delaySize = numWaveguides * maxDelay * sizeof(float);
    metal_->delayLines = metal_->device->newBuffer(delaySize, MTL::ResourceStorageModeShared);

    size_t paramsSize = numWaveguides * sizeof(WaveguideParams);
    metal_->paramsBuffer = metal_->device->newBuffer(paramsSize, MTL::ResourceStorageModeShared);

    metal_->filterState = metal_->device->newBuffer(numWaveguides * sizeof(float), MTL::ResourceStorageModeShared);
    metal_->writePositions = metal_->device->newBuffer(numWaveguides * sizeof(int), MTL::ResourceStorageModeShared);

    metal_->inputL = metal_->device->newBuffer(maxSamples * sizeof(float), MTL::ResourceStorageModeShared);
    metal_->inputR = metal_->device->newBuffer(maxSamples * sizeof(float), MTL::ResourceStorageModeShared);
    metal_->outputL = metal_->device->newBuffer(maxSamples * sizeof(float), MTL::ResourceStorageModeShared);
    metal_->outputR = metal_->device->newBuffer(maxSamples * sizeof(float), MTL::ResourceStorageModeShared);

    if (!metal_->delayLines || !metal_->paramsBuffer || !metal_->filterState ||
        !metal_->writePositions || !metal_->inputL || !metal_->inputR ||
        !metal_->outputL || !metal_->outputR) {
        error = "Failed to allocate GPU buffers";
        return false;
    }

    return true;
}

void WaveguideEngine::uploadParams() {
    if (!metal_->paramsBuffer || params_.empty()) return;

    std::memcpy(metal_->paramsBuffer->contents(), params_.data(),
                params_.size() * sizeof(WaveguideParams));
}

bool WaveguideEngine::loadPreset(const std::string& csvPath, std::string& error) {
    if (!WaveguideCSVLoader::loadFromFile(csvPath, params_, error,
                                          config_.maxDelayLineSamples)) {
        return false;
    }

    if (initialized_) {
        if (!allocateBuffers(error)) {
            return false;
        }
        uploadParams();
        reset();
    }

    return true;
}

void WaveguideEngine::generateRandomPreset(uint32_t seed) {
    WaveguideCSVLoader::generateRandom(params_, config_.maxWaveguides,
                                       config_.sampleRate,
                                       config_.maxDelayLineSamples,
                                       seed);

    if (initialized_) {
        std::string error;
        allocateBuffers(error);
        uploadParams();
        reset();
    }
}

void WaveguideEngine::processBlock(const float* const* inputs,
                                   int numInputChannels,
                                   float* const* outputs,
                                   int numOutputChannels,
                                   int numSamples) {
    if (!initialized_ || params_.empty() || numSamples == 0) {
        for (int ch = 0; ch < numOutputChannels; ch++) {
            std::memset(outputs[ch], 0, static_cast<size_t>(numSamples) * sizeof(float));
        }
        return;
    }

    const float* inL = (numInputChannels > 0 && inputs[0]) ? inputs[0] : nullptr;
    const float* inR = (numInputChannels > 1 && inputs[1]) ? inputs[1] : inL;
    float* outL = (numOutputChannels > 0) ? outputs[0] : nullptr;
    float* outR = (numOutputChannels > 1) ? outputs[1] : outL;

    int offset = 0;
    while (offset < numSamples) {
        int chunkSize = std::min(numSamples - offset, metal_->maxSamplesPerBlock);

        processChunk(inL ? inL + offset : nullptr,
                     inR ? inR + offset : nullptr,
                     outL ? outL + offset : nullptr,
                     outR ? outR + offset : nullptr,
                     chunkSize);

        offset += chunkSize;
    }
}

void WaveguideEngine::processChunk(const float* inL, const float* inR,
                                   float* outL, float* outR, int numSamples) {
    if (inL) {
        std::memcpy(metal_->inputL->contents(), inL, static_cast<size_t>(numSamples) * sizeof(float));
    } else {
        std::memset(metal_->inputL->contents(), 0, static_cast<size_t>(numSamples) * sizeof(float));
    }

    if (inR) {
        std::memcpy(metal_->inputR->contents(), inR, static_cast<size_t>(numSamples) * sizeof(float));
    } else {
        std::memcpy(metal_->inputR->contents(), metal_->inputL->contents(),
                    static_cast<size_t>(numSamples) * sizeof(float));
    }

    auto cmdBuffer = metal_->commandQueue->commandBuffer();
    if (!cmdBuffer) {
        if (outL) std::memset(outL, 0, static_cast<size_t>(numSamples) * sizeof(float));
        if (outR && outR != outL) std::memset(outR, 0, static_cast<size_t>(numSamples) * sizeof(float));
        return;
    }

    {
        auto encoder = cmdBuffer->computeCommandEncoder();
        encoder->setComputePipelineState(metal_->clearKernel);
        encoder->setBuffer(metal_->outputL, 0, 0);
        encoder->setBuffer(metal_->outputR, 0, 1);
        uint32_t ns = static_cast<uint32_t>(numSamples);
        encoder->setBytes(&ns, sizeof(ns), 2);

        MTL::Size gridSize = MTL::Size::Make(static_cast<NS::UInteger>(numSamples), 1, 1);
        NS::UInteger tgSize = static_cast<NS::UInteger>(std::min(64, numSamples));
        MTL::Size threadgroupSize = MTL::Size::Make(tgSize, 1, 1);
        encoder->dispatchThreads(gridSize, threadgroupSize);
        encoder->endEncoding();
    }

    {
        auto encoder = cmdBuffer->computeCommandEncoder();
        encoder->setComputePipelineState(metal_->processKernel);

        encoder->setBuffer(metal_->delayLines, 0, 0);
        encoder->setBuffer(metal_->paramsBuffer, 0, 1);
        encoder->setBuffer(metal_->inputL, 0, 2);
        encoder->setBuffer(metal_->inputR, 0, 3);
        encoder->setBuffer(metal_->outputL, 0, 4);
        encoder->setBuffer(metal_->outputR, 0, 5);
        encoder->setBuffer(metal_->filterState, 0, 6);
        encoder->setBuffer(metal_->writePositions, 0, 7);

        WaveguideUniforms uniforms;
        uniforms.numWaveguides = static_cast<uint32_t>(params_.size());
        uniforms.numSamples = static_cast<uint32_t>(numSamples);
        uniforms.maxDelayLength = static_cast<uint32_t>(config_.maxDelayLineSamples);
        uniforms.halfWaveguideCount = uniforms.numWaveguides / 2;
        encoder->setBytes(&uniforms, sizeof(uniforms), 8);

        NS::UInteger numWaveguides = static_cast<NS::UInteger>(params_.size());
        MTL::Size gridSize = MTL::Size::Make(numWaveguides, 1, 1);
        MTL::Size threadgroupSize = MTL::Size::Make(std::min<NS::UInteger>(64, numWaveguides), 1, 1);
        encoder->dispatchThreads(gridSize, threadgroupSize);
        encoder->endEncoding();
    }

    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();

    if (outL) {
        std::memcpy(outL, metal_->outputL->contents(),
                    static_cast<size_t>(numSamples) * sizeof(float));
    }
    if (outR && outR != outL) {
        std::memcpy(outR, metal_->outputR->contents(),
                    static_cast<size_t>(numSamples) * sizeof(float));
    }
}

void WaveguideEngine::reset() {
    if (!metal_->delayLines) return;

    size_t delaySize = params_.size() * static_cast<size_t>(config_.maxDelayLineSamples) * sizeof(float);
    std::memset(metal_->delayLines->contents(), 0, delaySize);
    std::memset(metal_->filterState->contents(), 0, params_.size() * sizeof(float));
    std::memset(metal_->writePositions->contents(), 0, params_.size() * sizeof(int));
}

void WaveguideEngine::shutdown() {
    initialized_ = false;
    metal_.reset();
    metal_ = std::make_unique<MetalState>();
    status_ = "Shutdown";
}

const char* WaveguideEngine::statusString() const {
    return status_.c_str();
}

}  // namespace ts::metal::waveguide
