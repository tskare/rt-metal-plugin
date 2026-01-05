#pragma once

#include <ts_metal_accel/third_party/concurrentqueue/concurrentqueue.h>
#include <ts_metal_accel/ts_metal_accel.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "FDTDCPU.h"
#include "FDTDTypes.h"

namespace MTL {
class Device;
class CommandQueue;
class Library;
class ComputePipelineState;
class Buffer;
}  // namespace MTL

namespace ts::metal::fdtd {

struct SourcePlacement {
  float x = 0.0f;  // Now supports fractional positions for better placement
  float y = 0.0f;
  float z = 0.0f;
  float gain = 1.0f;
};

struct MicPlacement {
  float x = 0.0f;  // Now supports fractional positions for interpolation
  float y = 0.0f;
  float z = 0.0f;
  float gain = 1.0f;
};

struct FDTDReverbConfig {
  SolverConfig solver;
  std::array<SourcePlacement, 2> sources{};
  std::array<MicPlacement, 2> mics{};
  float wetLevel = 1.0f;
  float dryLevel = 0.0f;
};

class FDTDReverbEngine {
 public:
  FDTDReverbEngine() = default;
  ~FDTDReverbEngine();

  bool prepare(const FDTDReverbConfig& newConfig, std::string& errorMessage);
  void reset();
  void shutdown();

  [[nodiscard]] bool isPrepared() const noexcept { return prepared; }
  [[nodiscard]] bool usesGPU() const noexcept {
    return preferGPU && gpuEnabled && gpu.ready;
  }
  [[nodiscard]] std::string statusString() const { return statusText; }

  void processBlock(const float* const* inputs, int numInputChannels,
                    float* const* outputs, int numOutputChannels,
                    int numSamples);

  void setExpectedBlockSize(int samples);

  void setGPUPreferred(bool preferGPUProcessing);
  [[nodiscard]] bool isGPUPreferred() const noexcept { return preferGPU; }
  void setWatchdogTimeout(std::chrono::nanoseconds timeout);
  [[nodiscard]] std::chrono::nanoseconds getWatchdogTimeout() const;

  struct AsyncMetrics {
    bool asyncEnabled = false;
    std::uint64_t completedFrames = 0;
    std::uint64_t underruns = 0;
    std::uint64_t watchdogTrips = 0;
    std::uint64_t warmupFramesRemaining = 0;
    double averageLatencyMillis = 0.0;
    double maxLatencyMillis = 0.0;
    std::size_t queueDepth = 0;
    double cpuEnqueueMillis = 0.0;
    double cpuCopyMillis = 0.0;
    double watchdogTimeoutMillis = 0.0;
    std::uint64_t processBlockCalls = 0;
    std::uint64_t enqueueAttempts = 0;
    std::uint64_t enqueueDrops = 0;
    std::uint64_t latencySampleCount = 0;
  };

  [[nodiscard]] AsyncMetrics getAsyncMetrics() const;

  [[nodiscard]] const FDTDReverbConfig& getConfig() const noexcept {
    return config;
  }

 private:
  [[nodiscard]] bool validateConfig(const FDTDReverbConfig& candidate,
                                    std::string& errorMessage) const;

  void processBlockCPU(const float* const* inputs, int numInputChannels,
                       float* const* outputs, int numOutputChannels,
                       int numSamples);

  void processBlockGPU(const float* const* inputs, int numInputChannels,
                       float* const* outputs, int numOutputChannels,
                       int numSamples);

  void injectSourcesCPU(const float* const* inputs, int numInputChannels,
                        int sampleIndex);
  [[nodiscard]] float samplePressureCPU(const MicPlacement& mic) const;

  std::uint32_t linearIndex(int x, int y, int z) const;

  bool initialiseGPU(std::string& errorMessage);
  void resetGPU();
  void shutdownGPU();

  FDTDReverbConfig config{};
  FDTDCPUSolver cpuSolver{};
  bool prepared = false;
  bool gpuEnabled = false;
  bool preferGPU = true;
  std::string statusText{"Uninitialised"};

  struct GPUState {
    bool ready = false;
    Uniforms uniforms{};
    std::uint32_t totalCells = 0;

    SourceCommandList sourceCommands;
    MicCommandList micCommands;

    ts::metal::MetalPtr<MTL::Device> device;
    ts::metal::MetalPtr<MTL::CommandQueue> commandQueue;
    ts::metal::MetalPtr<MTL::Library> library;
    ts::metal::MetalPtr<MTL::ComputePipelineState> processBlockPipeline;

    ts::metal::MetalPtr<MTL::Buffer> pressure[2];
    ts::metal::MetalPtr<MTL::Buffer> velocityX[2];
    ts::metal::MetalPtr<MTL::Buffer> velocityY[2];
    ts::metal::MetalPtr<MTL::Buffer> velocityZ[2];

    ts::metal::MetalPtr<MTL::Buffer> sourceCommandBuffer;
    ts::metal::MetalPtr<MTL::Buffer> micCommandBuffer;

    int activeIndex = 0;
    int parityForSubmit = 0;

    struct AsyncFrame {
      ts::metal::MetalPtr<MTL::Buffer> sourceSamples;
      std::size_t sourceCapacity = 0;
      ts::metal::MetalPtr<MTL::Buffer> micOutput;
      std::size_t micCapacity = 0;
      std::vector<float> micHostCopy;
      int numSamples = 0;
      int sourceCount = 0;
      int micCount = 0;
      int initialParity = 0;
      int finalParity = 0;
      std::uint64_t frameIndex = 0;
      bool hasError = false;
      std::string errorMessage;
      std::chrono::steady_clock::time_point submitTime;
      std::chrono::steady_clock::duration gpuDuration{};
      std::atomic<bool> inFlight{false};
    };

    struct AsyncState {
      bool enabled = false;
      std::atomic<bool> running{false};
      static constexpr std::size_t queueCapacity = 8;
      std::size_t activeFrameCount = 0;
      std::array<AsyncFrame, queueCapacity> frames{};
      ts::metal::SpscQueue<AsyncFrame*, queueCapacity> freeFrames;
      ts::metal::SpscQueue<AsyncFrame*, queueCapacity> jobQueue;
      moodycamel::ConcurrentQueue<AsyncFrame*> completedQueue;
      std::thread worker;
      std::mutex jobMutex;
      std::condition_variable jobCv;
      std::atomic<std::uint64_t> frameCounter{0};
      std::atomic<std::uint64_t> completedFrames{0};
      std::atomic<std::uint64_t> underrunCount{0};
      std::atomic<std::uint64_t> watchdogTrips{0};
      std::atomic<std::uint64_t> totalLatencyNs{0};
      std::atomic<std::uint64_t> maxLatencyNs{0};
      std::atomic<std::uint64_t> warmupFramesRemaining{0};
      std::atomic<std::int64_t> watchdogTimeoutNs{
          std::chrono::milliseconds(10).count()};
      int expectedBlockSize = 0;
      std::atomic<std::uint64_t> cpuEnqueueTimeNs{0};
      std::atomic<std::uint64_t> cpuEnqueueCount{0};
      std::atomic<std::uint64_t> cpuCopyTimeNs{0};
      std::atomic<std::uint64_t> cpuCopyCount{0};
      std::atomic<std::uint64_t> processBlockCalls{0};
      std::atomic<std::uint64_t> enqueueAttempts{0};
      std::atomic<std::uint64_t> enqueueDrops{0};
      std::atomic<std::uint64_t> latencySampleCount{0};
    } async;
  } gpu;

  bool configureAsyncState(std::string& errorMessage);
  void destroyAsyncState();
  void prepareFrameForReuse(GPUState::AsyncFrame& frame);
  bool ensureFrameCapacity(GPUState::AsyncFrame& frame, std::size_t sourceCount,
                           std::size_t micCount, int numSamples);
  void runAsyncWorker();
  bool encodeFrame(GPUState::AsyncFrame& frame);
  void pushGPUFailure(const std::string& message);
};

}  // namespace ts::metal::fdtd
