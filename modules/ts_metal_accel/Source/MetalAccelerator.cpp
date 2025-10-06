/*
 * ts-metal-accel - GPU-accelerated audio processing with Metal
 * Copyright (c) 2025 Travis Skare
 *
 * SPDX-License-Identifier: MIT
 */

#include "MetalAccelerator.h"

#include <Foundation/Foundation.hpp>
#include <algorithm>
#include <cstring>
#include <utility>

namespace ts {
namespace metal {

MetalAccelerator::MetalAccelerator() = default;

MetalAccelerator::~MetalAccelerator() { shutdown(); }

std::expected<void, MetalErrorInfo> MetalAccelerator::initialize() {
  if (initialized) return {};  // Already initialized - idempotent success

  // Initialize the shared Metal context
  auto result = MetalContext::getInstance().initialize();
  if (!result) return result;

  if (!getDevice() || !getCommandQueue()) {
    return std::unexpected(MetalErrorInfo{
        MetalError::NotInitialized, "MetalContext initialization failed"});
  }

  initialized = true;
  return {};
}

std::expected<void, MetalErrorInfo> MetalAccelerator::loadKernel(
    const std::string& kernelName, const std::string& mslSource) {
  AutoreleasePool pool;

  if (!initialized)
    return std::unexpected(MetalErrorInfo{MetalError::NotInitialized});

  auto* device = getDevice();
  if (!device)
    return std::unexpected(MetalErrorInfo{MetalError::DeviceNotFound});

  // Create NS::String from std::string
  NS::String* source =
      NS::String::string(mslSource.c_str(), NS::UTF8StringEncoding);
  if (!source) {
    return std::unexpected(
        MetalErrorInfo{MetalError::CompilationFailed,
                       "Failed to create NS::String from source"});
  }

  // Compile the Metal library
  NS::Error* error = nullptr;
  MTL::Library* library = device->newLibrary(source, nullptr, &error);

  if (!library || error) {
    std::string errorMsg = "Failed to compile Metal library";
    if (error && error->localizedDescription()) {
      errorMsg +=
          ": " + std::string(error->localizedDescription()->utf8String());
    }
    return std::unexpected(
        MetalErrorInfo{MetalError::CompilationFailed, errorMsg});
  }

  // Get the kernel function
  NS::String* funcName =
      NS::String::string(kernelName.c_str(), NS::UTF8StringEncoding);
  MTL::Function* function = library->newFunction(funcName);

  if (!function) {
    library->release();
    return std::unexpected(MetalErrorInfo{
        MetalError::InvalidKernel,
        "Kernel function '" + kernelName + "' not found in library"});
  }

  // Create compute pipeline state
  MTL::ComputePipelineState* pipelineState =
      device->newComputePipelineState(function, &error);

  if (!pipelineState || error) {
    std::string errorMsg = "Failed to create pipeline state";
    if (error && error->localizedDescription()) {
      errorMsg +=
          ": " + std::string(error->localizedDescription()->utf8String());
    }
    function->release();
    library->release();
    return std::unexpected(
        MetalErrorInfo{MetalError::CompilationFailed, errorMsg});
  }

  // Calculate optimal threadgroup size once at kernel load time
  // Use thread execution width (SIMD width, typically 32 on Apple GPUs)
  NS::UInteger threadExecutionWidth = pipelineState->threadExecutionWidth();
  NS::UInteger maxThreadsPerThreadgroup =
      pipelineState->maxTotalThreadsPerThreadgroup();

  // Use multiple of SIMD width for best performance, up to max supported
  NS::UInteger optimalSize = threadExecutionWidth;
  if (optimalSize == 0 || optimalSize > maxThreadsPerThreadgroup)
    optimalSize = maxThreadsPerThreadgroup;

  // Prefer 64 or 128 (multiples of 32) for better occupancy if supported
  if (maxThreadsPerThreadgroup >= 128)
    optimalSize = 128;
  else if (maxThreadsPerThreadgroup >= 64)
    optimalSize = 64;

  // Cache the kernel state with optimal threadgroup size
  KernelState state;
  state.library.reset(library);
  state.function.reset(function);
  state.pipelineState.reset(pipelineState);
  state.optimalThreadsPerThreadgroup = optimalSize;
  kernels[kernelName] = std::move(state);

  return {};
}

std::expected<void, MetalErrorInfo> MetalAccelerator::processBlock(
    const std::string& kernelName, juce::AudioBuffer<float>& buffer,
    const void* params, size_t paramsSize) {
  AutoreleasePool pool;

  if (!initialized)
    return std::unexpected(MetalErrorInfo{MetalError::NotInitialized});

  // Find the cached kernel
  auto it = kernels.find(kernelName);
  if (it == kernels.end()) {
    return std::unexpected(MetalErrorInfo{
        MetalError::InvalidKernel, "Kernel '" + kernelName + "' not loaded"});
  }

  auto& kernelState = it->second;
  auto* device = getDevice();
  auto* commandQueue = getCommandQueue();

  if (!device || !commandQueue)
    return std::unexpected(MetalErrorInfo{MetalError::DeviceNotFound});

  const int numChannels = buffer.getNumChannels();
  const int numSamples = buffer.getNumSamples();
  const size_t samplesPerChannel = static_cast<size_t>(numSamples);
  const size_t bufferSize =
      static_cast<size_t>(numChannels) * samplesPerChannel * sizeof(float);

  if (bufferSize == 0) return {};

  if (auto ensureResult = ensureBufferCapacity(bufferSize); !ensureResult)
    return ensureResult;

  // Copy channel-major data into shared Metal buffer
  auto* inputData = static_cast<float*>(inputBuffer->contents());
  for (int ch = 0; ch < numChannels; ++ch) {
    const float* channelData = buffer.getReadPointer(ch);
    std::memcpy(inputData + static_cast<size_t>(ch) * samplesPerChannel,
                channelData, samplesPerChannel * sizeof(float));
  }

  // Create command buffer
  MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
  if (!commandBuffer) {
    return std::unexpected(MetalErrorInfo{MetalError::ExecutionFailed,
                                          "Failed to create command buffer"});
  }

  // Create compute encoder
  MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
  if (!encoder) {
    return std::unexpected(MetalErrorInfo{MetalError::ExecutionFailed,
                                          "Failed to create compute encoder"});
  }

  // Set pipeline and buffers
  encoder->setComputePipelineState(kernelState.pipelineState.get());
  encoder->setBuffer(inputBuffer.get(), 0, 0);   // buffer(0)
  encoder->setBuffer(outputBuffer.get(), 0, 1);  // buffer(1)
  if (params && paramsSize > 0)
    encoder->setBytes(params, paramsSize, 2);  // buffer(2)

  // Pass buffer size for bounds checking in shader
  const uint32_t totalSamples = static_cast<uint32_t>(numChannels * numSamples);
  encoder->setBytes(&totalSamples, sizeof(totalSamples), 3);  // buffer(3)

  // Calculate grid size (one thread per sample)
  // Use cached optimal threadgroup size
  const NS::UInteger totalThreads = static_cast<NS::UInteger>(numChannels) *
                                    static_cast<NS::UInteger>(numSamples);
  NS::UInteger threadsPerThreadgroup = kernelState.optimalThreadsPerThreadgroup;

  // Clamp to actual work size if smaller than optimal
  threadsPerThreadgroup = std::min(threadsPerThreadgroup, totalThreads);
  threadsPerThreadgroup =
      std::max(static_cast<NS::UInteger>(1), threadsPerThreadgroup);

  const NS::UInteger threadgroupCount =
      (totalThreads + threadsPerThreadgroup - 1) / threadsPerThreadgroup;

  MTL::Size threadgroups = MTL::Size::Make(threadgroupCount, 1, 1);
  MTL::Size threadgroupSize = MTL::Size::Make(threadsPerThreadgroup, 1, 1);

  encoder->dispatchThreadgroups(threadgroups, threadgroupSize);
  encoder->endEncoding();

  // Commit and wait (BLOCKING - NOT REAL-TIME SAFE!)
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  // Copy results back to JUCE buffer
  const float* outputData = static_cast<const float*>(outputBuffer->contents());
  for (int ch = 0; ch < numChannels; ++ch) {
    float* channelData = buffer.getWritePointer(ch);
    std::memcpy(channelData,
                outputData + static_cast<size_t>(ch) * samplesPerChannel,
                samplesPerChannel * sizeof(float));
  }

  return {};
}

std::expected<void, MetalErrorInfo> MetalAccelerator::enableAsyncProcessing(
    const AsyncConfig& config) {
  if (!initialized)
    return std::unexpected(MetalErrorInfo{MetalError::NotInitialized});

  if (config.maxChannels <= 0 || config.maxSamplesPerBlock <= 0 ||
      config.queueDepth == 0)
    return std::unexpected(MetalErrorInfo{MetalError::InvalidBuffer,
                                          "Invalid async configuration"});

  if (config.queueDepth >= kAsyncQueueCapacity)
    return std::unexpected(MetalErrorInfo{
        MetalError::InvalidBuffer, "Async queue depth exceeds capacity"});

  // Validate configuration against reasonable limits
  if (config.maxChannels > kMaxReasonableChannels) {
    return std::unexpected(MetalErrorInfo{
        MetalError::InvalidBuffer, "maxChannels " +
                                       std::to_string(config.maxChannels) +
                                       " exceeds reasonable limit of " +
                                       std::to_string(kMaxReasonableChannels)});
  }

  if (config.maxSamplesPerBlock > kMaxReasonableBlockSize) {
    return std::unexpected(MetalErrorInfo{
        MetalError::InvalidBuffer,
        "maxSamplesPerBlock " + std::to_string(config.maxSamplesPerBlock) +
            " exceeds reasonable limit of " +
            std::to_string(kMaxReasonableBlockSize)});
  }

  disableAsyncProcessing();

  auto* device = getDevice();
  if (!device)
    return std::unexpected(MetalErrorInfo{MetalError::DeviceNotFound});

  const size_t bufferBytes = static_cast<size_t>(config.maxChannels) *
                             static_cast<size_t>(config.maxSamplesPerBlock) *
                             sizeof(float);

  if (bufferBytes == 0)
    return std::unexpected(
        MetalErrorInfo{MetalError::InvalidBuffer, "Async buffer size is zero"});

  async.config = config;
  async.bufferCapacityBytes = bufferBytes;
  async.activeFrameCount = static_cast<size_t>(config.queueDepth);

  for (size_t i = 0; i < async.frames.size(); ++i) {
    prepareFrameForReuse(async.frames[i]);

    if (i < async.activeFrameCount) {
      auto input = makeMetalPtr(
          device->newBuffer(bufferBytes, MTL::ResourceStorageModeShared));
      if (!input) {
        disableAsyncProcessing();
        return std::unexpected(
            MetalErrorInfo{MetalError::InvalidBuffer,
                           "Failed to allocate async input buffer"});
      }

      auto output = makeMetalPtr(
          device->newBuffer(bufferBytes, MTL::ResourceStorageModeShared));
      if (!output) {
        disableAsyncProcessing();
        return std::unexpected(
            MetalErrorInfo{MetalError::InvalidBuffer,
                           "Failed to allocate async output buffer"});
      }

      async.frames[i].input = std::move(input);
      async.frames[i].output = std::move(output);
      async.frames[i].paramStorage.resize(config.maxParameterBytes);
    } else {
      async.frames[i].input.reset();
      async.frames[i].output.reset();
      async.frames[i].paramStorage.clear();
    }
  }

  async.freeFrames.reset();
  async.jobQueue.reset();
  async.completedQueue = decltype(async.completedQueue)();

  for (size_t i = 0; i < async.activeFrameCount; ++i) {
    if (!async.freeFrames.push(&async.frames[i])) {
      disableAsyncProcessing();
      return std::unexpected(MetalErrorInfo{
          MetalError::ExecutionFailed, "Failed to seed async frame queue"});
    }
  }

  async.frameCounter.store(0, std::memory_order_relaxed);
  async.totalFrames.store(0, std::memory_order_relaxed);
  async.underrunCount.store(0, std::memory_order_relaxed);
  const auto warmupFrames =
      std::max(kAsyncWarmupFrames, async.config.queueDepth);
  async.warmupFramesRemaining.store(warmupFrames, std::memory_order_relaxed);
  async.enabled = true;
  async.running.store(true, std::memory_order_release);
  async.worker = std::thread([this]() { runAsyncWorker(); });

  return {};
}

void MetalAccelerator::disableAsyncProcessing() { stopAsyncProcessing(); }

std::expected<void, MetalErrorInfo> MetalAccelerator::processBlockAsync(
    const std::string& kernelName, juce::AudioBuffer<float>& buffer,
    const void* params, size_t paramsSize) {
  if (!async.enabled)
    return std::unexpected(MetalErrorInfo{MetalError::NotInitialized,
                                          "Async processing not enabled"});

  if (!initialized)
    return std::unexpected(MetalErrorInfo{MetalError::NotInitialized});

  auto warmupRemaining =
      async.warmupFramesRemaining.load(std::memory_order_relaxed);
  const bool inWarmup = warmupRemaining > 0;
  if (inWarmup) {
    async.warmupFramesRemaining.store(warmupRemaining - 1,
                                      std::memory_order_relaxed);
  } else {
    async.totalFrames.fetch_add(1, std::memory_order_relaxed);
  }

  // Handle completed GPU work from previous blocks first
  FrameResources* completedFrame = nullptr;
  if (async.completedQueue.try_dequeue(completedFrame)) {
    if (completedFrame->hasError) {
      auto error = std::move(completedFrame->error);
      prepareFrameForReuse(*completedFrame);
      async.freeFrames.push(completedFrame);
      return std::unexpected(std::move(error));
    }

    const size_t srcSamples = static_cast<size_t>(completedFrame->numSamples);
    const size_t dstSamples = static_cast<size_t>(buffer.getNumSamples());
    const size_t samplesToCopy = std::min(srcSamples, dstSamples);
    const int channelsToCopy =
        std::min(buffer.getNumChannels(), completedFrame->numChannels);

    const float* outputData =
        static_cast<const float*>(completedFrame->output->contents());
    for (int ch = 0; ch < channelsToCopy; ++ch) {
      std::memcpy(buffer.getWritePointer(ch),
                  outputData + static_cast<size_t>(ch) * srcSamples,
                  samplesToCopy * sizeof(float));

      if (dstSamples > samplesToCopy) {
        auto* writePtr = buffer.getWritePointer(ch);
        std::fill(writePtr + samplesToCopy, writePtr + dstSamples, 0.0f);
      }
    }

    for (int ch = channelsToCopy; ch < buffer.getNumChannels(); ++ch)
      buffer.clear(ch, 0, buffer.getNumSamples());

    prepareFrameForReuse(*completedFrame);
    async.freeFrames.push(completedFrame);
  } else {
    // Queue underrun: no completed frame available
    // Clear buffer to silence to avoid outputting garbage or stale audio
    if (!inWarmup) async.underrunCount.fetch_add(1, std::memory_order_relaxed);
    buffer.clear();
  }

  FrameResources* frame = nullptr;
  if (!async.freeFrames.pop(frame)) {
    return std::unexpected(MetalErrorInfo{MetalError::ExecutionFailed,
                                          "No free async frames available"});
  }

  auto kernelIt = kernels.find(kernelName);
  if (kernelIt == kernels.end()) {
    async.freeFrames.push(frame);
    return std::unexpected(
        MetalErrorInfo{MetalError::InvalidKernel, "Kernel not loaded"});
  }

  frame->kernel = &kernelIt->second;
  frame->numChannels = buffer.getNumChannels();
  frame->numSamples = buffer.getNumSamples();

  if (frame->numChannels > async.config.maxChannels) {
    async.freeFrames.push(frame);
    return std::unexpected(
        MetalErrorInfo{MetalError::InvalidBuffer,
                       "Channel count exceeds async configuration"});
  }

  const size_t samplesPerChannel = static_cast<size_t>(frame->numSamples);
  const size_t bufferBytes = static_cast<size_t>(frame->numChannels) *
                             samplesPerChannel * sizeof(float);

  if (bufferBytes > async.bufferCapacityBytes) {
    async.freeFrames.push(frame);
    return std::unexpected(MetalErrorInfo{
        MetalError::InvalidBuffer, "Block size exceeds async configuration"});
  }

  if (params && paramsSize > 0) {
    if (paramsSize > async.config.maxParameterBytes) {
      async.freeFrames.push(frame);
      return std::unexpected(
          MetalErrorInfo{MetalError::InvalidBuffer,
                         "Parameter payload exceeds async configuration"});
    }

    if (!frame->paramStorage.empty())
      std::memcpy(frame->paramStorage.data(), params, paramsSize);
    frame->paramSize = paramsSize;
  } else {
    frame->paramSize = 0;
  }

  auto* inputData = static_cast<float*>(frame->input->contents());
  for (int ch = 0; ch < frame->numChannels; ++ch) {
    const float* channelData = buffer.getReadPointer(ch);
    std::memcpy(inputData + static_cast<size_t>(ch) * samplesPerChannel,
                channelData, samplesPerChannel * sizeof(float));
  }

  frame->frameIndex =
      async.frameCounter.fetch_add(1, std::memory_order_relaxed);

  if (!async.jobQueue.push(frame)) {
    async.freeFrames.push(frame);
    return std::unexpected(
        MetalErrorInfo{MetalError::ExecutionFailed, "Async queue full"});
  }

  async.jobCv.notify_one();

  return {};
}

void MetalAccelerator::shutdown() {
  disableAsyncProcessing();

  kernels.clear();
  initialized = false;
  inputBuffer.reset();
  outputBuffer.reset();
  bufferCapacityBytes = 0;
}

MTL::Device* MetalAccelerator::getDevice() const {
  return MetalContext::getInstance().getDevice();
}

MTL::CommandQueue* MetalAccelerator::getCommandQueue() const {
  return MetalContext::getInstance().getCommandQueue();
}

MetalAccelerator::GPUStats MetalAccelerator::getGPUStats() const {
  GPUStats stats{};
  stats.asyncEnabled = async.enabled;

  if (async.enabled) {
    stats.totalFrames = async.totalFrames.load(std::memory_order_relaxed);
    stats.underrunCount = async.underrunCount.load(std::memory_order_relaxed);
    stats.warmupFramesRemaining = static_cast<uint64_t>(
        async.warmupFramesRemaining.load(std::memory_order_relaxed));
    stats.underrunRate = stats.totalFrames > 0
                             ? static_cast<double>(stats.underrunCount) /
                                   static_cast<double>(stats.totalFrames)
                             : 0.0;
  }

  return stats;
}

std::expected<void, MetalErrorInfo> MetalAccelerator::ensureBufferCapacity(
    size_t requiredBytes) {
  // Validate against maximum reasonable size
  if (requiredBytes > kMaxBufferBytes) {
    return std::unexpected(
        MetalErrorInfo{MetalError::InvalidBuffer,
                       "Buffer size " + std::to_string(requiredBytes) +
                           " exceeds maximum limit of " +
                           std::to_string(kMaxBufferBytes) + " bytes"});
  }

  if (bufferCapacityBytes >= requiredBytes && inputBuffer && outputBuffer)
    return {};

  // WARNING: This allocation happens on the calling thread
  // For sync mode, this is the audio thread (NOT real-time safe!)
  // For async mode, buffers are pre-allocated during enableAsyncProcessing

  auto* device = getDevice();
  if (!device)
    return std::unexpected(MetalErrorInfo{MetalError::DeviceNotFound});

  // Allocate to requested size (for sync mode) or max size (for async mode
  // pre-allocation)
  auto newInput = makeMetalPtr(
      device->newBuffer(requiredBytes, MTL::ResourceStorageModeShared));
  if (!newInput) {
    return std::unexpected(MetalErrorInfo{MetalError::InvalidBuffer,
                                          "Failed to allocate input buffer"});
  }

  auto newOutput = makeMetalPtr(
      device->newBuffer(requiredBytes, MTL::ResourceStorageModeShared));
  if (!newOutput) {
    return std::unexpected(MetalErrorInfo{MetalError::InvalidBuffer,
                                          "Failed to allocate output buffer"});
  }

  inputBuffer = std::move(newInput);
  outputBuffer = std::move(newOutput);
  bufferCapacityBytes = requiredBytes;
  return {};
}

void MetalAccelerator::stopAsyncProcessing() {
  bool wasRunning = async.running.exchange(false, std::memory_order_acq_rel);

  if (wasRunning) {
    async.jobCv.notify_all();
    if (async.worker.joinable()) async.worker.join();
    async.worker = std::thread();
  }

  async.enabled = false;
  async.freeFrames.reset();
  async.jobQueue.reset();

  FrameResources* drainedFrame = nullptr;
  while (async.completedQueue.try_dequeue(drainedFrame)) {
    if (drainedFrame != nullptr) prepareFrameForReuse(*drainedFrame);
  }
  async.completedQueue = decltype(async.completedQueue)();

  for (auto& frame : async.frames) {
    frame.input.reset();
    frame.output.reset();
    frame.paramStorage.clear();
    prepareFrameForReuse(frame);
  }

  async.bufferCapacityBytes = 0;
  async.activeFrameCount = 0;
  async.frameCounter.store(0, std::memory_order_relaxed);
  async.totalFrames.store(0, std::memory_order_relaxed);
  async.underrunCount.store(0, std::memory_order_relaxed);
  async.warmupFramesRemaining.store(0, std::memory_order_relaxed);
  async.config = {};
}

void MetalAccelerator::runAsyncWorker() {
  while (true) {
    FrameResources* frame = nullptr;
    if (!async.jobQueue.pop(frame)) {
      std::unique_lock<std::mutex> lock(async.jobMutex);
      // Use wait_for with timeout to ensure we don't deadlock on shutdown
      async.jobCv.wait_for(lock, std::chrono::milliseconds(100), [this]() {
        return !async.running.load(std::memory_order_acquire) ||
               !async.jobQueue.empty();
      });

      if (!async.running.load(std::memory_order_acquire) &&
          async.jobQueue.empty())
        break;

      continue;
    }

    if (frame == nullptr) continue;

    // Submit GPU work - encodeFrame now returns immediately after submitting
    // The completion handler will push the frame to completedQueue when GPU
    // finishes
    auto encodeResult = encodeFrame(*frame);
    if (!encodeResult) {
      // Encoding failed before GPU submission - push error to completed queue
      // immediately
      frame->hasError = true;
      frame->error = encodeResult.error();

      async.completedQueue.enqueue(frame);
    }
    // If encoding succeeded, the completion handler will handle pushing to
    // completedQueue
  }
}

void MetalAccelerator::prepareFrameForReuse(FrameResources& frame) {
  frame.paramSize = 0;
  frame.numChannels = 0;
  frame.numSamples = 0;
  frame.kernel = nullptr;
  frame.frameIndex = 0;
  frame.hasError = false;
  frame.error = MetalErrorInfo{MetalError::Unknown};
}

std::expected<void, MetalErrorInfo> MetalAccelerator::encodeFrame(
    FrameResources& frame) {
  AutoreleasePool pool;

  if (frame.kernel == nullptr)
    return std::unexpected(MetalErrorInfo{MetalError::InvalidKernel,
                                          "Async frame missing kernel"});

  auto* commandQueue = getCommandQueue();
  if (!commandQueue)
    return std::unexpected(MetalErrorInfo{MetalError::DeviceNotFound,
                                          "Command queue unavailable"});

  if (frame.numChannels == 0 || frame.numSamples == 0) return {};

  MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
  if (!commandBuffer)
    return std::unexpected(MetalErrorInfo{MetalError::ExecutionFailed,
                                          "Failed to create command buffer"});

  MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
  if (!encoder)
    return std::unexpected(MetalErrorInfo{MetalError::ExecutionFailed,
                                          "Failed to create compute encoder"});

  const auto& kernelState = *frame.kernel;

  encoder->setComputePipelineState(kernelState.pipelineState.get());
  encoder->setBuffer(frame.input.get(), 0, 0);
  encoder->setBuffer(frame.output.get(), 0, 1);
  if (frame.paramSize > 0 && !frame.paramStorage.empty())
    encoder->setBytes(frame.paramStorage.data(), frame.paramSize, 2);

  // Pass buffer size for bounds checking in shader
  const uint32_t totalSamples =
      static_cast<uint32_t>(frame.numChannels * frame.numSamples);
  encoder->setBytes(&totalSamples, sizeof(totalSamples), 3);  // buffer(3)

  // Use cached optimal threadgroup size
  const NS::UInteger totalThreads =
      static_cast<NS::UInteger>(frame.numChannels) *
      static_cast<NS::UInteger>(frame.numSamples);
  NS::UInteger threadsPerThreadgroup = kernelState.optimalThreadsPerThreadgroup;

  // Clamp to actual work size if smaller than optimal
  threadsPerThreadgroup = std::min(threadsPerThreadgroup, totalThreads);
  threadsPerThreadgroup =
      std::max(static_cast<NS::UInteger>(1), threadsPerThreadgroup);

  const NS::UInteger threadgroupCount =
      (totalThreads + threadsPerThreadgroup - 1) / threadsPerThreadgroup;

  MTL::Size threadgroups = MTL::Size::Make(threadgroupCount, 1, 1);
  MTL::Size threadgroupSize = MTL::Size::Make(threadsPerThreadgroup, 1, 1);

  encoder->dispatchThreadgroups(threadgroups, threadgroupSize);
  encoder->endEncoding();

  // Use completion handler instead of blocking - this allows the worker thread
  // to continue processing other frames while the GPU executes asynchronously
  FrameResources* framePtr = &frame;
  commandBuffer->addCompletedHandler(^(MTL::CommandBuffer* cb) {
    // This handler is called by Metal on an arbitrary thread when GPU work
    // completes
    if (cb->status() != MTL::CommandBufferStatusCompleted) {
      // GPU execution failed
      framePtr->hasError = true;
      framePtr->error =
          MetalErrorInfo{MetalError::ExecutionFailed,
                         "GPU command buffer failed with status: " +
                             std::to_string(static_cast<int>(cb->status()))};
    }

    // Push completed frame to queue (thread-safe, multi-producer)
    async.completedQueue.enqueue(framePtr);
  });

  commandBuffer->commit();
  // Return immediately - GPU work happens asynchronously

  return {};
}

}  // namespace metal
}  // namespace ts
