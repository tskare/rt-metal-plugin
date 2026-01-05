#include "FDTDReverbEngine.h"

#include <ts_metal_accel/Source/MetalContext.h>
#include <ts_metal_accel/Source/MetalUtils.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "FDTDMetal.h"

namespace ts::metal::fdtd {
namespace {
constexpr int maxChannels = 2;

inline int clampToGrid(float coord, int extent) {
  if (extent <= 0) return 0;

  const int rounded = static_cast<int>(std::lround(coord));
  return std::clamp(rounded, 0, extent - 1);
}

inline bool inBounds(const SolverConfig& solver, float x, float y, float z) {
  return x >= 0.0f && x < static_cast<float>(solver.grid.nx) && y >= 0.0f &&
         y < static_cast<float>(solver.grid.ny) && z >= 0.0f &&
         z < static_cast<float>(solver.grid.nz);
}

ts::metal::MetalPtr<MTL::ComputePipelineState> buildPipeline(
    MTL::Device* device, MTL::Library* library, const char* functionName,
    std::string& errorMessage) {
  ts::metal::MetalPtr<MTL::ComputePipelineState> pipeline;
  NS::Error* error = nullptr;

  if (auto* fnName = NS::String::string(functionName, NS::UTF8StringEncoding)) {
    if (auto* function = library->newFunction(fnName)) {
      pipeline.reset(device->newComputePipelineState(function, &error));
      function->release();
    }
  }

  if (pipeline.get() == nullptr || error != nullptr) {
    std::string message = "Failed to create pipeline for ";
    message += functionName;

    if (error != nullptr && error->localizedDescription() != nullptr)
      message +=
          ": " + std::string(error->localizedDescription()->utf8String());

    errorMessage = message;
  }
  return pipeline;
}

inline void dispatch1D(MTL::ComputeCommandEncoder* encoder,
                       NS::UInteger items) {
  if (items == 0) return;

  const NS::UInteger threadsPerGroup = std::min<NS::UInteger>(64, items);
  const NS::UInteger groups = (items + threadsPerGroup - 1) / threadsPerGroup;
  encoder->dispatchThreadgroups(MTL::Size::Make(groups, 1, 1),
                                MTL::Size::Make(threadsPerGroup, 1, 1));
}
}  // namespace

bool FDTDReverbEngine::validateConfig(const FDTDReverbConfig& candidate,
                                      std::string& errorMessage) const {
  try {
    FDTDCPUSolver temp{candidate.solver};
    (void)temp;
  } catch (const std::exception& e) {
    errorMessage = e.what();
    return false;
  }

  for (const auto& source : candidate.sources) {
    if (!inBounds(candidate.solver, source.x, source.y, source.z)) {
      errorMessage = "Source position out of bounds";
      return false;
    }
  }

  for (const auto& mic : candidate.mics) {
    if (!inBounds(candidate.solver, mic.x, mic.y, mic.z)) {
      errorMessage = "Mic position out of bounds";
      return false;
    }
  }

  return true;
}

bool FDTDReverbEngine::prepare(const FDTDReverbConfig& newConfig,
                               std::string& errorMessage) {
  if (!validateConfig(newConfig, errorMessage)) return false;

  shutdownGPU();

  config = newConfig;
  cpuSolver.configure(config.solver);
  cpuSolver.reset();

  gpuEnabled = initialiseGPU(errorMessage);

  const std::string dims = std::to_string(config.solver.grid.nx) + "x" +
                           std::to_string(config.solver.grid.ny) + "x" +
                           std::to_string(config.solver.grid.nz);

  if (!gpuEnabled) {
    statusText = "FDTD CPU fallback (" + dims + ")";
    if (!errorMessage.empty()) statusText += " - " + errorMessage;
    errorMessage.clear();
  } else if (!preferGPU) {
    statusText = "FDTD CPU (forced)";
    errorMessage.clear();
  } else {
    statusText = "FDTD GPU (" + dims + ")";
    errorMessage.clear();
  }

  prepared = true;
  return true;
}

FDTDReverbEngine::~FDTDReverbEngine() { shutdown(); }

void FDTDReverbEngine::reset() {
  cpuSolver.reset();
  if (gpu.ready) resetGPU();

  prepared = true;
}

void FDTDReverbEngine::shutdown() {
  shutdownGPU();
  cpuSolver.reset();
  gpuEnabled = false;
  prepared = false;
}

void FDTDReverbEngine::processBlock(const float* const* inputs,
                                    int numInputChannels, float* const* outputs,
                                    int numOutputChannels, int numSamples) {
  if (!prepared) return;

  if (preferGPU && gpuEnabled && gpu.ready)
    processBlockGPU(inputs, numInputChannels, outputs, numOutputChannels,
                    numSamples);
  else
    processBlockCPU(inputs, numInputChannels, outputs, numOutputChannels,
                    numSamples);
}

void FDTDReverbEngine::processBlockCPU(const float* const* inputs,
                                       int numInputChannels,
                                       float* const* outputs,
                                       int numOutputChannels, int numSamples) {
  const int outputChannels = std::min(maxChannels, numOutputChannels);

  for (int ch = 0; ch < outputChannels; ++ch)
    std::fill(outputs[ch], outputs[ch] + numSamples, 0.0f);

  for (int sample = 0; sample < numSamples; ++sample) {
    injectSourcesCPU(inputs, numInputChannels, sample);
    cpuSolver.step();

    for (int ch = 0; ch < outputChannels; ++ch) {
      const auto& mic = config.mics[static_cast<std::size_t>(ch)];
      const float wet = samplePressureCPU(mic) * config.wetLevel;
      outputs[ch][sample] += wet;
    }
  }

  const float dryLevel = config.dryLevel;
  if (dryLevel > 0.0f) {
    const int copyChannels =
        std::min(maxChannels, std::min(numInputChannels, numOutputChannels));
    for (int ch = 0; ch < copyChannels; ++ch) {
      if (inputs[ch] == nullptr) continue;

      const float* src = inputs[ch];
      float* dst = outputs[ch];
      for (int sample = 0; sample < numSamples; ++sample)
        dst[sample] += dryLevel * src[sample];
    }
  }
}

void FDTDReverbEngine::processBlockGPU(const float* const* inputs,
                                       int numInputChannels,
                                       float* const* outputs,
                                       int numOutputChannels, int numSamples) {
  // Audio thread responsibilities:
  //   1. Drain one completed GPU frame (if available) and copy its wet mix into
  //   the output block.
  //   2. Add the dry signal for the current block.
  //   3. Stage the next block's source data and enqueue it to the worker thread
  //   so the GPU can run off-thread.
  // Each enqueue adds one block of latency; the worker drives Metal command
  // buffers via completion handlers.
  const int outputChannels = std::min(maxChannels, numOutputChannels);

  const int inputChannels = std::min(maxChannels, numInputChannels);

  std::vector<float> inputSnapshot(static_cast<std::size_t>(inputChannels) *
                                   static_cast<std::size_t>(numSamples));
  std::array<const float*, maxChannels> snapshotPtrs{};

  for (int ch = 0; ch < inputChannels; ++ch) {
    float* dest =
        inputSnapshot.data() +
        static_cast<std::size_t>(ch) * static_cast<std::size_t>(numSamples);
    if (inputs[ch] != nullptr)
      std::memcpy(dest, inputs[ch],
                  static_cast<std::size_t>(numSamples) * sizeof(float));
    else
      std::fill(dest, dest + numSamples, 0.0f);

    snapshotPtrs[static_cast<std::size_t>(ch)] = dest;
  }

  for (int ch = inputChannels; ch < maxChannels; ++ch)
    snapshotPtrs[static_cast<std::size_t>(ch)] = nullptr;

  auto clearOutputs = [&]() {
    for (int ch = 0; ch < outputChannels; ++ch)
      std::fill(outputs[ch], outputs[ch] + numSamples, 0.0f);
  };

  auto renderDryOnly = [&]() {
    const float dryLevel = config.dryLevel;
    if (dryLevel <= 0.0f) return;

    const int copyChannels =
        std::min(maxChannels, std::min(numInputChannels, numOutputChannels));
    for (int ch = 0; ch < copyChannels; ++ch) {
      const auto* src = snapshotPtrs[static_cast<std::size_t>(ch)];
      if (src == nullptr) continue;

      float* dst = outputs[ch];
      for (int sample = 0; sample < numSamples; ++sample)
        dst[sample] += dryLevel * src[sample];
    }
  };

  clearOutputs();

  if (!gpu.ready || !gpu.processBlockPipeline || gpu.device.get() == nullptr ||
      gpu.commandQueue.get() == nullptr) {
    renderDryOnly();
    return;
  }

  auto& async = gpu.async;
  GPUState::AsyncFrame* completedFrame = nullptr;
  const bool asyncActive =
      async.enabled && async.running.load(std::memory_order_acquire);

  if (asyncActive && async.completedQueue.try_dequeue(completedFrame)) {
    if (completedFrame != nullptr) {
      if (completedFrame->hasError) {
        const std::string message = completedFrame->errorMessage.empty()
                                        ? "GPU command buffer failed"
                                        : completedFrame->errorMessage;

        prepareFrameForReuse(*completedFrame);
        async.freeFrames.push(completedFrame);
        pushGPUFailure(message);
        renderDryOnly();
        return;
      }

      const auto completedSamples =
          static_cast<std::size_t>(completedFrame->numSamples);
      const auto completedMics =
          static_cast<std::size_t>(completedFrame->micCount);
      const auto micStride =
          completedMics == 0 ? std::size_t{1} : completedMics;
      const auto copyMics = std::min<std::size_t>(
          completedMics, static_cast<std::size_t>(outputChannels));
      const auto copySamples = std::min<std::size_t>(
          completedSamples, static_cast<std::size_t>(numSamples));

      if (!completedFrame->micHostCopy.empty() && copySamples > 0 &&
          copyMics > 0) {
        const auto copyStart = std::chrono::steady_clock::now();
        for (std::size_t sample = 0; sample < copySamples; ++sample) {
          for (std::size_t mic = 0; mic < copyMics; ++mic) {
            const float wet =
                completedFrame->micHostCopy[sample * micStride + mic] *
                config.wetLevel;
            outputs[static_cast<int>(mic)][static_cast<int>(sample)] += wet;
          }
        }
        const auto copyElapsed = std::chrono::steady_clock::now() - copyStart;
        async.cpuCopyTimeNs.fetch_add(
            static_cast<std::uint64_t>(copyElapsed.count()),
            std::memory_order_relaxed);
        async.cpuCopyCount.fetch_add(1, std::memory_order_relaxed);
      }

      gpu.activeIndex = completedFrame->finalParity & 1;

      const auto warmupRemaining =
          async.warmupFramesRemaining.load(std::memory_order_relaxed);
      bool includeInStats = (warmupRemaining == 0);
      if (warmupRemaining > 0) {
        includeInStats = (warmupRemaining == 1);
        async.warmupFramesRemaining.store(warmupRemaining - 1,
                                          std::memory_order_relaxed);
      }

      const auto gpuDuration = completedFrame->gpuDuration.count();
      async.completedFrames.fetch_add(1, std::memory_order_relaxed);

      if (includeInStats) {
        async.latencySampleCount.fetch_add(1, std::memory_order_relaxed);
        async.totalLatencyNs.fetch_add(static_cast<std::uint64_t>(gpuDuration),
                                       std::memory_order_relaxed);

        std::uint64_t prevMax =
            async.maxLatencyNs.load(std::memory_order_relaxed);
        while (static_cast<std::uint64_t>(gpuDuration) > prevMax &&
               !async.maxLatencyNs.compare_exchange_weak(
                   prevMax, static_cast<std::uint64_t>(gpuDuration),
                   std::memory_order_relaxed, std::memory_order_relaxed)) {
        }
      }

      prepareFrameForReuse(*completedFrame);
      async.freeFrames.push(completedFrame);
      completedFrame = nullptr;
    }
  } else if (asyncActive) {
    const auto warmupRemaining =
        async.warmupFramesRemaining.load(std::memory_order_relaxed);
    if (warmupRemaining == 0) {
      bool triggered = false;
      const auto watchdogNs =
          async.watchdogTimeoutNs.load(std::memory_order_relaxed);
      const auto now = std::chrono::steady_clock::now();

      if (watchdogNs > 0) {
        const auto watchdogTimeout = std::chrono::nanoseconds(watchdogNs);
        for (std::size_t i = 0; i < async.activeFrameCount; ++i) {
          auto& candidate = async.frames[i];
          if (candidate.inFlight.load(std::memory_order_acquire) &&
              candidate.submitTime.time_since_epoch().count() != 0) {
            if (now - candidate.submitTime > watchdogTimeout) {
              triggered = true;
              break;
            }
          }
        }
      }

      if (triggered) {
        async.watchdogTrips.fetch_add(1, std::memory_order_relaxed);
        pushGPUFailure("GPU watchdog timeout");
        renderDryOnly();
        return;
      }

      async.underrunCount.fetch_add(1, std::memory_order_relaxed);
    }
  }

  renderDryOnly();

  if (!asyncActive || !gpu.ready) return;

  const std::size_t sourceCount = gpu.sourceCommands.size();
  const std::size_t micCount = gpu.micCommands.size();

  const NS::UInteger totalCells = static_cast<NS::UInteger>(gpu.totalCells);
  if (totalCells == 0) return;

  async.processBlockCalls.fetch_add(1, std::memory_order_relaxed);

  GPUState::AsyncFrame* frame = nullptr;
  if (!async.freeFrames.pop(frame) || frame == nullptr) {
    async.underrunCount.fetch_add(1, std::memory_order_relaxed);
    pushGPUFailure("GPU frame queue exhausted");
    renderDryOnly();
    return;
  }

  const auto enqueueStart = std::chrono::steady_clock::now();
  async.enqueueAttempts.fetch_add(1, std::memory_order_relaxed);

  if (!ensureFrameCapacity(*frame, sourceCount, micCount, numSamples)) {
    prepareFrameForReuse(*frame);
    async.freeFrames.push(frame);
    pushGPUFailure("Failed to allocate async buffers");
    renderDryOnly();
    return;
  }

  frame->numSamples = numSamples;
  frame->sourceCount = static_cast<int>(sourceCount);
  frame->micCount = static_cast<int>(micCount);
  frame->initialParity = gpu.parityForSubmit & 1;
  frame->finalParity = frame->initialParity ^ (numSamples & 1);
  frame->frameIndex =
      async.frameCounter.fetch_add(1, std::memory_order_relaxed);
  frame->hasError = false;
  frame->errorMessage.clear();

  const std::size_t sourceStride = sourceCount == 0 ? 1 : sourceCount;
  if (frame->sourceSamples) {
    auto* sourcePtr = static_cast<float*>(frame->sourceSamples->contents());
    if (sourcePtr != nullptr) {
      for (int sample = 0; sample < numSamples; ++sample) {
        for (std::size_t s = 0; s < sourceCount; ++s) {
          float value = 0.0f;
          if (s < static_cast<std::size_t>(inputChannels)) {
            const auto* src = snapshotPtrs[static_cast<std::size_t>(s)];
            if (src != nullptr) value = src[sample];
          }
          value *= config.sources[s].gain;
          sourcePtr[static_cast<std::size_t>(sample) * sourceStride + s] =
              value;
        }
      }
    }
  }

  if (frame->micOutput) {
    const std::size_t micBytes = std::max<std::size_t>(micCount, 1) *
                                 static_cast<std::size_t>(numSamples) *
                                 sizeof(float);
    std::memset(frame->micOutput->contents(), 0, micBytes);
  }

  const auto requiredHostSamples =
      micCount * static_cast<std::size_t>(numSamples);
  frame->micHostCopy.resize(requiredHostSamples);

  frame->submitTime = std::chrono::steady_clock::now();
  frame->inFlight.store(true, std::memory_order_release);

  if (!async.jobQueue.push(frame)) {
    frame->inFlight.store(false, std::memory_order_release);
    prepareFrameForReuse(*frame);
    async.freeFrames.push(frame);
    async.underrunCount.fetch_add(1, std::memory_order_relaxed);
    async.enqueueDrops.fetch_add(1, std::memory_order_relaxed);
    pushGPUFailure("Failed to enqueue GPU work");
    renderDryOnly();
    return;
  }

  async.jobCv.notify_one();
  gpu.parityForSubmit = frame->finalParity & 1;

  const auto enqueueElapsed = std::chrono::steady_clock::now() - enqueueStart;
  async.cpuEnqueueTimeNs.fetch_add(
      static_cast<std::uint64_t>(enqueueElapsed.count()),
      std::memory_order_relaxed);
  async.cpuEnqueueCount.fetch_add(1, std::memory_order_relaxed);
}

void FDTDReverbEngine::setExpectedBlockSize(int samples) {
  const int clamped = std::max(samples, 0);
  gpu.async.expectedBlockSize = clamped;

  if (!gpu.ready || !gpu.async.enabled) return;

  const std::size_t sourceCount = gpu.sourceCommands.size();
  const std::size_t micCount = gpu.micCommands.size();

  for (std::size_t i = 0; i < gpu.async.activeFrameCount; ++i) {
    if (!ensureFrameCapacity(gpu.async.frames[i], sourceCount, micCount,
                             clamped)) {
      pushGPUFailure("Failed to preallocate GPU buffers");
      break;
    }
  }
}

void FDTDReverbEngine::setGPUPreferred(bool preferGPUProcessing) {
  preferGPU = preferGPUProcessing;

  if (!preferGPU) {
    statusText = "FDTD CPU (forced)";
    return;
  }

  if (gpuEnabled && gpu.ready) {
    const std::string dims = std::to_string(config.solver.grid.nx) + "x" +
                             std::to_string(config.solver.grid.ny) + "x" +
                             std::to_string(config.solver.grid.nz);
    statusText = "FDTD GPU (" + dims + ")";
  }
}

void FDTDReverbEngine::setWatchdogTimeout(std::chrono::nanoseconds timeout) {
  gpu.async.watchdogTimeoutNs.store(timeout.count(), std::memory_order_relaxed);
}

std::chrono::nanoseconds FDTDReverbEngine::getWatchdogTimeout() const {
  return std::chrono::nanoseconds(
      gpu.async.watchdogTimeoutNs.load(std::memory_order_relaxed));
}

FDTDReverbEngine::AsyncMetrics FDTDReverbEngine::getAsyncMetrics() const {
  AsyncMetrics metrics;
  metrics.asyncEnabled =
      gpu.async.enabled && gpu.async.running.load(std::memory_order_relaxed);
  metrics.completedFrames =
      gpu.async.completedFrames.load(std::memory_order_relaxed);
  metrics.underruns = gpu.async.underrunCount.load(std::memory_order_relaxed);
  metrics.watchdogTrips =
      gpu.async.watchdogTrips.load(std::memory_order_relaxed);
  metrics.warmupFramesRemaining =
      gpu.async.warmupFramesRemaining.load(std::memory_order_relaxed);
  metrics.queueDepth = gpu.async.activeFrameCount;

  const auto totalLatency =
      gpu.async.totalLatencyNs.load(std::memory_order_relaxed);
  const auto maxLatency =
      gpu.async.maxLatencyNs.load(std::memory_order_relaxed);
  const auto cpuEnqueueNs =
      gpu.async.cpuEnqueueTimeNs.load(std::memory_order_relaxed);
  const auto cpuEnqueueCount =
      gpu.async.cpuEnqueueCount.load(std::memory_order_relaxed);
  const auto cpuCopyNs =
      gpu.async.cpuCopyTimeNs.load(std::memory_order_relaxed);
  const auto cpuCopyCount =
      gpu.async.cpuCopyCount.load(std::memory_order_relaxed);
  const auto watchdogNs =
      gpu.async.watchdogTimeoutNs.load(std::memory_order_relaxed);
  const auto latencySamples =
      gpu.async.latencySampleCount.load(std::memory_order_relaxed);

  metrics.latencySampleCount = latencySamples;

  if (latencySamples > 0)
    metrics.averageLatencyMillis = static_cast<double>(totalLatency) /
                                   static_cast<double>(latencySamples) /
                                   1000000.0;
  metrics.maxLatencyMillis = static_cast<double>(maxLatency) / 1000000.0;

  if (cpuEnqueueCount > 0)
    metrics.cpuEnqueueMillis = static_cast<double>(cpuEnqueueNs) /
                               static_cast<double>(cpuEnqueueCount) / 1000000.0;
  if (cpuCopyCount > 0)
    metrics.cpuCopyMillis = static_cast<double>(cpuCopyNs) /
                            static_cast<double>(cpuCopyCount) / 1000000.0;
  metrics.watchdogTimeoutMillis =
      watchdogNs > 0 ? static_cast<double>(watchdogNs) / 1000000.0 : 0.0;
  metrics.processBlockCalls =
      gpu.async.processBlockCalls.load(std::memory_order_relaxed);
  metrics.enqueueAttempts =
      gpu.async.enqueueAttempts.load(std::memory_order_relaxed);
  metrics.enqueueDrops = gpu.async.enqueueDrops.load(std::memory_order_relaxed);

  return metrics;
}

void FDTDReverbEngine::injectSourcesCPU(const float* const* inputs,
                                        int numInputChannels, int sampleIndex) {
  const int channels = std::min(maxChannels, numInputChannels);
  for (int ch = 0; ch < channels; ++ch) {
    const auto& source = config.sources[static_cast<std::size_t>(ch)];
    const float inputSample = inputs[ch][sampleIndex];
    const int ix = clampToGrid(source.x, config.solver.grid.nx);
    const int iy = clampToGrid(source.y, config.solver.grid.ny);
    const int iz = clampToGrid(source.z, config.solver.grid.nz);
    cpuSolver.addPressureImpulse(ix, iy, iz, inputSample * source.gain);
  }
}

float FDTDReverbEngine::samplePressureCPU(const MicPlacement& mic) const {
  return cpuSolver.pressureAtInterpolated(mic.x, mic.y, mic.z) * mic.gain;
}

std::uint32_t FDTDReverbEngine::linearIndex(int x, int y, int z) const {
  const int nx = config.solver.grid.nx;
  const int ny = config.solver.grid.ny;
  const int nz = config.solver.grid.nz;

  const int clampedX = std::clamp(x, 0, nx - 1);
  const int clampedY = std::clamp(y, 0, ny - 1);
  const int clampedZ = std::clamp(z, 0, nz - 1);

  return static_cast<std::uint32_t>(clampedX) +
         static_cast<std::uint32_t>(nx) *
             (static_cast<std::uint32_t>(clampedY) +
              static_cast<std::uint32_t>(ny) *
                  static_cast<std::uint32_t>(clampedZ));
}

bool FDTDReverbEngine::initialiseGPU(std::string& errorMessage) {
  gpu.ready = false;
  gpu.sourceCommands.clear();
  gpu.micCommands.clear();

  auto& context = MetalContext::getInstance();
  if (auto initResult = context.initialize(); !initResult) {
    errorMessage = "Metal context initialisation failed";
    if (!initResult.error().message.empty())
      errorMessage += ": " + initResult.error().message;
    return false;
  }

  auto* device = context.getDevice();
  auto* commandQueue = context.getCommandQueue();
  if (device == nullptr || commandQueue == nullptr) {
    errorMessage = "Metal device or command queue unavailable";
    return false;
  }

  gpu.device = retainMetalPtr(device);
  gpu.commandQueue = retainMetalPtr(commandQueue);

  ts::metal::AutoreleasePool pool;

  if (auto* source = NS::String::string(getMetalKernelSource().c_str(),
                                        NS::UTF8StringEncoding)) {
    NS::Error* error = nullptr;
    gpu.library.reset(device->newLibrary(source, nullptr, &error));
    if (error != nullptr) {
      errorMessage = "Failed to compile FDTD kernels: " +
                     std::string(error->localizedDescription()->utf8String());
      return false;
    }
  }

  if (!gpu.library) {
    errorMessage = "Failed to create Metal library";
    return false;
  }

  gpu.processBlockPipeline = buildPipeline(device, gpu.library.get(),
                                           "fdtd_process_block", errorMessage);
  if (!gpu.processBlockPipeline) return false;

  gpu.uniforms.grid.nx = static_cast<std::uint32_t>(config.solver.grid.nx);
  gpu.uniforms.grid.ny = static_cast<std::uint32_t>(config.solver.grid.ny);
  gpu.uniforms.grid.nz = static_cast<std::uint32_t>(config.solver.grid.nz);
  gpu.uniforms.coeffVelocity =
      config.solver.dt / (config.solver.density * config.solver.dx);
  const float c = config.solver.soundSpeed;
  gpu.uniforms.coeffPressure =
      config.solver.density * c * c * config.solver.dt / config.solver.dx;
  gpu.uniforms.boundaryAttenuation = config.solver.boundaryAttenuation;

  gpu.totalCells =
      gpu.uniforms.grid.nx * gpu.uniforms.grid.ny * gpu.uniforms.grid.nz;
  const std::size_t bufferBytes =
      static_cast<std::size_t>(gpu.totalCells) * sizeof(float);

  for (int i = 0; i < 2; ++i) {
    gpu.pressure[i] = makeMetalPtr(
        device->newBuffer(bufferBytes, MTL::ResourceStorageModeShared));
    gpu.velocityX[i] = makeMetalPtr(
        device->newBuffer(bufferBytes, MTL::ResourceStorageModeShared));
    gpu.velocityY[i] = makeMetalPtr(
        device->newBuffer(bufferBytes, MTL::ResourceStorageModeShared));
    gpu.velocityZ[i] = makeMetalPtr(
        device->newBuffer(bufferBytes, MTL::ResourceStorageModeShared));

    if (!gpu.pressure[i] || !gpu.velocityX[i] || !gpu.velocityY[i] ||
        !gpu.velocityZ[i]) {
      errorMessage = "Failed to allocate FDTD state buffers";
      return false;
    }
  }

  gpu.sourceCommands.resize(config.sources.size());
  for (std::size_t i = 0; i < gpu.sourceCommands.size(); ++i) {
    const auto& src = config.sources[i];
    const int ix = clampToGrid(src.x, config.solver.grid.nx);
    const int iy = clampToGrid(src.y, config.solver.grid.ny);
    const int iz = clampToGrid(src.z, config.solver.grid.nz);
    gpu.sourceCommands[i].index = linearIndex(ix, iy, iz);
  }

  gpu.micCommands.resize(config.mics.size());
  for (std::size_t i = 0; i < gpu.micCommands.size(); ++i) {
    const auto& mic = config.mics[i];
    gpu.micCommands[i] = {mic.x, mic.y, mic.z, mic.gain};
  }

  const std::size_t sourceBufferBytes =
      std::max<std::size_t>(gpu.sourceCommands.size(), 1) *
      sizeof(SourceCommand);
  const std::size_t micBufferBytes =
      std::max<std::size_t>(gpu.micCommands.size(), 1) * sizeof(MicCommand);

  gpu.sourceCommandBuffer = makeMetalPtr(
      device->newBuffer(sourceBufferBytes, MTL::ResourceStorageModeShared));
  gpu.micCommandBuffer = makeMetalPtr(
      device->newBuffer(micBufferBytes, MTL::ResourceStorageModeShared));

  if (!gpu.sourceCommandBuffer || !gpu.micCommandBuffer) {
    errorMessage = "Failed to allocate command buffers";
    return false;
  }

  if (!gpu.sourceCommands.empty())
    std::memcpy(gpu.sourceCommandBuffer->contents(), gpu.sourceCommands.data(),
                gpu.sourceCommands.size() * sizeof(SourceCommand));

  if (!gpu.micCommands.empty())
    std::memcpy(gpu.micCommandBuffer->contents(), gpu.micCommands.data(),
                gpu.micCommands.size() * sizeof(MicCommand));

  resetGPU();

  if (!configureAsyncState(errorMessage)) return false;

  gpu.activeIndex = 0;
  gpu.ready = true;
  return true;
}

void FDTDReverbEngine::resetGPU() {
  if (gpu.totalCells == 0) return;

  const std::size_t bufferBytes =
      static_cast<std::size_t>(gpu.totalCells) * sizeof(float);
  for (int i = 0; i < 2; ++i) {
    if (gpu.pressure[i])
      std::memset(gpu.pressure[i]->contents(), 0, bufferBytes);
    if (gpu.velocityX[i])
      std::memset(gpu.velocityX[i]->contents(), 0, bufferBytes);
    if (gpu.velocityY[i])
      std::memset(gpu.velocityY[i]->contents(), 0, bufferBytes);
    if (gpu.velocityZ[i])
      std::memset(gpu.velocityZ[i]->contents(), 0, bufferBytes);
  }
  gpu.activeIndex = 0;
  gpu.parityForSubmit = 0;

  if (gpu.async.enabled) {
    const auto warmup =
        std::max<std::size_t>(gpu.async.activeFrameCount * 2, 2);
    gpu.async.warmupFramesRemaining.store(warmup, std::memory_order_relaxed);
  }
}

void FDTDReverbEngine::shutdownGPU() {
  gpu.ready = false;
  destroyAsyncState();
  gpu.sourceCommands.clear();
  gpu.micCommands.clear();
  gpu.sourceCommandBuffer.reset();
  gpu.micCommandBuffer.reset();

  for (int i = 0; i < 2; ++i) {
    gpu.pressure[i].reset();
    gpu.velocityX[i].reset();
    gpu.velocityY[i].reset();
    gpu.velocityZ[i].reset();
  }

  gpu.processBlockPipeline.reset();
  gpu.library.reset();
  gpu.commandQueue.reset();
  gpu.device.reset();
  gpu.totalCells = 0;
}

bool FDTDReverbEngine::configureAsyncState(std::string& errorMessage) {
  destroyAsyncState();

  if (!gpu.device || !gpu.commandQueue || !gpu.processBlockPipeline) {
    errorMessage = "GPU context unavailable";
    return false;
  }

  auto& async = gpu.async;
  const auto cellCount = static_cast<std::size_t>(gpu.totalCells);
  std::size_t desiredFrames = 3;
  if (cellCount > 32768) desiredFrames = 4;
  if (cellCount > 65536) desiredFrames = 5;
  if (cellCount > 98304) desiredFrames = 6;
  async.activeFrameCount =
      std::clamp<std::size_t>(desiredFrames, 2, async.queueCapacity - 1);

  async.freeFrames.reset();
  async.jobQueue.reset();
  async.completedQueue = decltype(async.completedQueue)();

  for (auto& frame : async.frames) prepareFrameForReuse(frame);

  const std::size_t sourceCount = gpu.sourceCommands.size();
  const std::size_t micCount = gpu.micCommands.size();
  if (async.expectedBlockSize > 0) {
    for (std::size_t i = 0; i < async.activeFrameCount; ++i) {
      if (!ensureFrameCapacity(async.frames[i], sourceCount, micCount,
                               async.expectedBlockSize)) {
        errorMessage = "Failed to preallocate async buffers";
        destroyAsyncState();
        return false;
      }
    }
  }

  for (std::size_t i = 0; i < async.activeFrameCount; ++i) {
    if (!async.freeFrames.push(&async.frames[i])) {
      errorMessage = "Failed to seed async frame queue";
      destroyAsyncState();
      return false;
    }
  }

  async.frameCounter.store(0, std::memory_order_relaxed);
  async.completedFrames.store(0, std::memory_order_relaxed);
  async.underrunCount.store(0, std::memory_order_relaxed);
  async.watchdogTrips.store(0, std::memory_order_relaxed);
  async.totalLatencyNs.store(0, std::memory_order_relaxed);
  async.maxLatencyNs.store(0, std::memory_order_relaxed);
  const auto warmupFrames =
      std::max<std::size_t>(async.activeFrameCount * 2, 2);
  async.warmupFramesRemaining.store(warmupFrames, std::memory_order_relaxed);
  async.cpuEnqueueTimeNs.store(0, std::memory_order_relaxed);
  async.cpuEnqueueCount.store(0, std::memory_order_relaxed);
  async.cpuCopyTimeNs.store(0, std::memory_order_relaxed);
  async.cpuCopyCount.store(0, std::memory_order_relaxed);
  async.processBlockCalls.store(0, std::memory_order_relaxed);
  async.enqueueAttempts.store(0, std::memory_order_relaxed);
  async.enqueueDrops.store(0, std::memory_order_relaxed);
  async.latencySampleCount.store(0, std::memory_order_relaxed);
  async.running.store(true, std::memory_order_release);
  async.enabled = true;

  try {
    async.worker = std::thread([this]() { runAsyncWorker(); });
  } catch (const std::exception& e) {
    async.running.store(false, std::memory_order_release);
    async.enabled = false;
    errorMessage = "Failed to start GPU worker: ";
    errorMessage += e.what();
    return false;
  } catch (...) {
    async.running.store(false, std::memory_order_release);
    async.enabled = false;
    errorMessage = "Failed to start GPU worker (unknown error)";
    return false;
  }

  gpu.parityForSubmit = gpu.activeIndex & 1;
  return true;
}

void FDTDReverbEngine::destroyAsyncState() {
  auto& async = gpu.async;
  const bool wasRunning =
      async.running.exchange(false, std::memory_order_acq_rel);

  if (wasRunning) {
    async.jobCv.notify_all();
    if (async.worker.joinable()) async.worker.join();
    async.worker = std::thread();
  }

  GPUState::AsyncFrame* drained = nullptr;
  while (async.completedQueue.try_dequeue(drained)) {
    if (drained != nullptr) prepareFrameForReuse(*drained);
  }
  async.completedQueue = decltype(async.completedQueue)();

  async.freeFrames.reset();
  async.jobQueue.reset();

  for (auto& frame : async.frames) {
    frame.sourceSamples.reset();
    frame.sourceCapacity = 0;
    frame.micOutput.reset();
    frame.micCapacity = 0;
    frame.micHostCopy.clear();
    frame.micHostCopy.shrink_to_fit();
    prepareFrameForReuse(frame);
  }

  async.activeFrameCount = 0;
  async.enabled = false;
}

void FDTDReverbEngine::prepareFrameForReuse(GPUState::AsyncFrame& frame) {
  frame.numSamples = 0;
  frame.sourceCount = 0;
  frame.micCount = 0;
  frame.initialParity = 0;
  frame.finalParity = 0;
  frame.frameIndex = 0;
  frame.hasError = false;
  frame.errorMessage.clear();
  frame.submitTime = {};
  frame.gpuDuration = {};
  frame.inFlight.store(false, std::memory_order_relaxed);
}

bool FDTDReverbEngine::ensureFrameCapacity(GPUState::AsyncFrame& frame,
                                           std::size_t sourceCount,
                                           std::size_t micCount,
                                           int numSamples) {
  if (!gpu.device) return false;

  const std::size_t samples = static_cast<std::size_t>(std::max(numSamples, 0));
  const std::size_t requiredSourceBytes =
      std::max<std::size_t>(sourceCount, 1) * samples * sizeof(float);
  const std::size_t requiredMicBytes =
      std::max<std::size_t>(micCount, 1) * samples * sizeof(float);

  if (requiredSourceBytes > frame.sourceCapacity) {
    auto buffer = makeMetalPtr(gpu.device->newBuffer(
        std::max<std::size_t>(requiredSourceBytes, sizeof(float)),
        MTL::ResourceStorageModeShared));
    if (!buffer) return false;
    frame.sourceSamples = std::move(buffer);
    frame.sourceCapacity = requiredSourceBytes;
  }

  if (requiredMicBytes > frame.micCapacity) {
    auto buffer = makeMetalPtr(gpu.device->newBuffer(
        std::max<std::size_t>(requiredMicBytes, sizeof(float)),
        MTL::ResourceStorageModeShared));
    if (!buffer) return false;
    frame.micOutput = std::move(buffer);
    frame.micCapacity = requiredMicBytes;
  }

  const std::size_t requiredHostSamples = micCount * samples;
  if (frame.micHostCopy.capacity() < requiredHostSamples)
    frame.micHostCopy.reserve(requiredHostSamples);

  return true;
}

void FDTDReverbEngine::runAsyncWorker() {
  while (gpu.async.running.load(std::memory_order_acquire)) {
    GPUState::AsyncFrame* frame = nullptr;
    if (!gpu.async.jobQueue.pop(frame)) {
      std::unique_lock<std::mutex> lock(gpu.async.jobMutex);
      gpu.async.jobCv.wait_for(lock, std::chrono::milliseconds(2), [this]() {
        return !gpu.async.running.load(std::memory_order_acquire) ||
               !gpu.async.jobQueue.empty();
      });
      continue;
    }

    if (frame == nullptr) continue;

    if (!encodeFrame(*frame)) {
      if (frame->errorMessage.empty())
        frame->errorMessage = "Failed to encode GPU frame";
      frame->hasError = true;
      frame->inFlight.store(false, std::memory_order_release);
      gpu.async.completedQueue.enqueue(frame);
    }
  }
}

bool FDTDReverbEngine::encodeFrame(GPUState::AsyncFrame& frame) {
  ts::metal::AutoreleasePool pool;

  auto* commandQueue = gpu.commandQueue.get();
  if (commandQueue == nullptr) {
    frame.errorMessage = "Command queue unavailable";
    return false;
  }

  if (!gpu.processBlockPipeline) {
    frame.errorMessage = "Pipeline not initialised";
    return false;
  }

  MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
  if (commandBuffer == nullptr) {
    frame.errorMessage = "Failed to create command buffer";
    return false;
  }

  MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
  if (encoder == nullptr) {
    frame.errorMessage = "Failed to create compute encoder";
    return false;
  }

  const NS::UInteger totalCells = static_cast<NS::UInteger>(gpu.totalCells);
  const NS::UInteger maxThreads = 1024;
  const NS::UInteger threadsPerGroup = std::max<NS::UInteger>(
      1, std::min<NS::UInteger>(maxThreads, totalCells == 0 ? 1 : totalCells));

  encoder->setComputePipelineState(gpu.processBlockPipeline.get());
  encoder->setBuffer(gpu.pressure[0].get(), 0, 0);
  encoder->setBuffer(gpu.pressure[1].get(), 0, 1);
  encoder->setBuffer(gpu.velocityX[0].get(), 0, 2);
  encoder->setBuffer(gpu.velocityX[1].get(), 0, 3);
  encoder->setBuffer(gpu.velocityY[0].get(), 0, 4);
  encoder->setBuffer(gpu.velocityY[1].get(), 0, 5);
  encoder->setBuffer(gpu.velocityZ[0].get(), 0, 6);
  encoder->setBuffer(gpu.velocityZ[1].get(), 0, 7);
  encoder->setBuffer(frame.sourceSamples ? frame.sourceSamples.get() : nullptr,
                     0, 8);
  encoder->setBuffer(gpu.sourceCommandBuffer.get(), 0, 9);
  encoder->setBuffer(gpu.micCommandBuffer.get(), 0, 10);
  encoder->setBuffer(frame.micOutput ? frame.micOutput.get() : nullptr, 0, 11);
  encoder->setBytes(&gpu.uniforms, sizeof(gpu.uniforms), 12);

  const uint32_t numSamples32 = static_cast<uint32_t>(frame.numSamples);
  const uint32_t sourceCount32 =
      static_cast<uint32_t>(std::max(frame.sourceCount, 0));
  const uint32_t micCount32 =
      static_cast<uint32_t>(std::max(frame.micCount, 0));
  const uint32_t parity32 = static_cast<uint32_t>(frame.initialParity & 1);

  encoder->setBytes(&numSamples32, sizeof(numSamples32), 13);
  encoder->setBytes(&sourceCount32, sizeof(sourceCount32), 14);
  encoder->setBytes(&micCount32, sizeof(micCount32), 15);
  encoder->setBytes(&parity32, sizeof(parity32), 16);
  encoder->dispatchThreadgroups(MTL::Size::Make(1, 1, 1),
                                MTL::Size::Make(threadsPerGroup, 1, 1));
  encoder->endEncoding();

  GPUState::AsyncFrame* framePtr = &frame;
  commandBuffer->addCompletedHandler([this, framePtr](MTL::CommandBuffer* cb) {
    ts::metal::AutoreleasePool completionPool;

    if (cb->status() != MTL::CommandBufferStatusCompleted) {
      framePtr->hasError = true;
      framePtr->errorMessage = "GPU command buffer failed with status: " +
                               std::to_string(static_cast<int>(cb->status()));
    } else {
      const std::size_t micSamples =
          static_cast<std::size_t>(framePtr->micCount) *
          static_cast<std::size_t>(framePtr->numSamples);
      if (micSamples > 0 && framePtr->micOutput &&
          !framePtr->micHostCopy.empty()) {
        const auto bytes = micSamples * sizeof(float);
        std::memcpy(framePtr->micHostCopy.data(),
                    framePtr->micOutput->contents(), bytes);
      }
      framePtr->hasError = false;
    }

    framePtr->gpuDuration =
        std::chrono::steady_clock::now() - framePtr->submitTime;
    framePtr->inFlight.store(false, std::memory_order_release);
    gpu.async.completedQueue.enqueue(framePtr);
  });

  commandBuffer->commit();
  return true;
}

void FDTDReverbEngine::pushGPUFailure(const std::string& message) {
  if (preferGPU) {
    statusText = "FDTD GPU warning: " + message;
    return;
  }

  statusText = "FDTD GPU fallback: " + message;
  gpu.ready = false;
  gpuEnabled = false;
  cpuSolver.reset();
}

}  // namespace ts::metal::fdtd
