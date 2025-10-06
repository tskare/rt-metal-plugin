/*
 * ts-metal-accel - GPU-accelerated audio processing with Metal
 * Copyright (c) 2025 Travis Skare
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <juce_audio_basics/juce_audio_basics.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "../third_party/concurrentqueue/concurrentqueue.h"
#include "MetalContext.h"
#include "MetalUtils.h"

namespace ts {
namespace metal {

/**
 * Phase 1: Synchronous GPU-accelerated audio processor.
 *
 * WARNING: This Phase 1 implementation is NOT real-time safe!
 * The processBlock() method calls waitUntilCompleted(), which blocks the audio
 * thread until GPU execution finishes. This can cause audio dropouts and should
 * only be used for development, testing, and validation.
 *
 * Phase 2 will implement asynchronous processing with lock-free queues.
 *
 * Lifecycle:
 *   1. Construction (lightweight)
 *   2. initialize() - sets up Metal connection (call from prepareToPlay)
 *   3. loadKernel() - compile shaders (call from prepareToPlay)
 *   4. processBlock() - execute GPU processing (call from processBlock)
 *   5. shutdown() - cleanup (call from releaseResources)
 *
 * Example usage:
 *   MetalAccelerator accel;
 *   if (auto result = accel.initialize(); !result) {
 *       // handle error
 *   }
 *   if (auto result = accel.loadKernel("distortion", mslSource); !result) {
 *       // handle error
 *   }
 *   // Later, in processBlock:
 *   if (auto result = accel.processBlock("distortion", buffer, params);
 * !result) {
 *       // handle error
 *   }
 */
class MetalAccelerator {
 public:
  MetalAccelerator();
  ~MetalAccelerator();

  /**
   * Initialize Metal environment.
   * Thread-safe, idempotent. Call from non-realtime context (prepareToPlay).
   */
  std::expected<void, MetalErrorInfo> initialize();

  /**
   * Compile and cache a Metal compute kernel.
   *
   * @param kernelName The name of the kernel function in the MSL source
   * @param mslSource The Metal Shading Language source code
   * @return Success or error information
   *
   * Caches the compiled pipeline state for future use.
   * Call from non-realtime context (prepareToPlay).
   */
  std::expected<void, MetalErrorInfo> loadKernel(const std::string& kernelName,
                                                 const std::string& mslSource);

  /**
   * Process an audio buffer on the GPU (PHASE 1: BLOCKING/SYNCHRONOUS).
   *
   * WARNING: This blocks the audio thread waiting for GPU completion!
   *
   * @param kernelName The name of the previously loaded kernel
   * @param buffer JUCE audio buffer to process (in-place)
   * @param params Optional kernel parameters (passed as constant buffer)
   * @return Success or error information
   */
  std::expected<void, MetalErrorInfo> processBlock(
      const std::string& kernelName, juce::AudioBuffer<float>& buffer,
      const void* params = nullptr, size_t paramsSize = 0);

  struct AsyncConfig {
    int maxChannels = 2;
    int maxSamplesPerBlock = 512;
    size_t queueDepth = 3;
    size_t maxParameterBytes = 256;
  };

  /**
   * GPU performance statistics for monitoring async queue health.
   */
  struct GPUStats {
    uint64_t totalFrames = 0;    // Total frames processed in async mode
    uint64_t underrunCount = 0;  // Times the completed queue was empty
    double underrunRate = 0.0;   // Underrun percentage (0.0 - 1.0)
    bool asyncEnabled = false;   // Whether async mode is active
    uint64_t warmupFramesRemaining =
        0;  // Frames left before metrics become active
  };

  /**
   * Enable Phase 2 asynchronous processing. Must be called from a
   * non-realtime thread (prepareToPlay). Replaces any previous async config.
   */
  std::expected<void, MetalErrorInfo> enableAsyncProcessing(
      const AsyncConfig& config);

  /**
   * Disable async processing and join the worker thread.
   */
  void disableAsyncProcessing();

  /**
   * Phase 2 processing path for the audio thread. Returns immediately
   * while the GPU executes on a worker thread. Adds one block of latency.
   */
  std::expected<void, MetalErrorInfo> processBlockAsync(
      const std::string& kernelName, juce::AudioBuffer<float>& buffer,
      const void* params = nullptr, size_t paramsSize = 0);

  /**
   * Shutdown and release all Metal resources.
   * Call from non-realtime context (releaseResources or destructor).
   */
  void shutdown();

  /**
   * Check if the accelerator is initialized and ready.
   */
  bool isInitialized() const { return initialized; }

  bool isAsyncEnabled() const { return async.enabled; }

  /**
   * Get current GPU performance statistics.
   * Thread-safe, lightweight read of atomic counters.
   */
  GPUStats getGPUStats() const;

 private:
  struct KernelState {
    MetalPtr<MTL::Library> library;
    MetalPtr<MTL::Function> function;
    MetalPtr<MTL::ComputePipelineState> pipelineState;
    NS::UInteger optimalThreadsPerThreadgroup = 0;  // Cached threadgroup size
  };

  bool initialized = false;
  std::map<std::string, KernelState> kernels;

  // Helper to get device/queue from context
  MTL::Device* getDevice() const;
  MTL::CommandQueue* getCommandQueue() const;

  std::expected<void, MetalErrorInfo> ensureBufferCapacity(
      size_t requiredBytes);

  MetalPtr<MTL::Buffer> inputBuffer;
  MetalPtr<MTL::Buffer> outputBuffer;
  size_t bufferCapacityBytes = 0;

  // Maximum buffer size limits to prevent unbounded allocation
  static constexpr int kMaxReasonableChannels = 32;
  static constexpr int kMaxReasonableBlockSize = 8192;
  static constexpr size_t kMaxBufferBytes =
      kMaxReasonableChannels * kMaxReasonableBlockSize * sizeof(float);

  static constexpr size_t kAsyncQueueCapacity = 8;
  static constexpr size_t kAsyncWarmupFrames = 12;

  struct FrameResources {
    MetalPtr<MTL::Buffer> input;
    MetalPtr<MTL::Buffer> output;
    std::vector<std::byte> paramStorage;
    size_t paramSize = 0;
    int numChannels = 0;
    int numSamples = 0;
    const KernelState* kernel = nullptr;
    uint64_t frameIndex = 0;
    bool hasError = false;
    MetalErrorInfo error{MetalError::Unknown};
  };

  struct AsyncState {
    bool enabled = false;
    std::atomic<bool> running{false};
    AsyncConfig config;
    size_t bufferCapacityBytes = 0;
    size_t activeFrameCount = 0;
    std::array<FrameResources, kAsyncQueueCapacity> frames;
    SpscQueue<FrameResources*, kAsyncQueueCapacity> freeFrames;
    SpscQueue<FrameResources*, kAsyncQueueCapacity> jobQueue;
    moodycamel::ConcurrentQueue<FrameResources*> completedQueue;
    std::thread worker;
    std::mutex jobMutex;
    std::condition_variable jobCv;
    std::atomic<uint64_t> frameCounter{0};
    // Queue underrun detection
    std::atomic<uint64_t> underrunCount{0};
    std::atomic<uint64_t> totalFrames{0};
    std::atomic<size_t> warmupFramesRemaining{0};
  } async;

  void stopAsyncProcessing();
  void runAsyncWorker();
  void prepareFrameForReuse(FrameResources& frame);
  std::expected<void, MetalErrorInfo> encodeFrame(FrameResources& frame);
};

}  // namespace metal
}  // namespace ts
