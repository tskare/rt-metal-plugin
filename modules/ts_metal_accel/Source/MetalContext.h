/*
 * ts-metal-accel - GPU-accelerated audio processing with Metal
 * Copyright (c) 2025 Travis Skare
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <mutex>

#include "MetalUtils.h"

namespace ts {
namespace metal {

/**
 * Singleton managing shared Metal resources (Device and CommandQueue).
 *
 * These core objects are expensive to create and should be shared across
 * all MetalAccelerator instances in an application.
 *
 * Thread-safe initialization guarded by a mutex so the context can be
 * torn down and rebuilt safely.
 */
class MetalContext {
 public:
  // Get the singleton instance
  static MetalContext& getInstance();

  // Initialize Metal resources (idempotent, thread-safe)
  std::expected<void, MetalErrorInfo> initialize();

  // Get the shared device
  MTL::Device* getDevice() const { return device.get(); }

  // Get the shared command queue
  MTL::CommandQueue* getCommandQueue() const { return commandQueue.get(); }

  // Check if initialized
  bool isInitialized() const { return device && commandQueue; }

  // Explicitly shutdown (called from destructor, or manually for cleanup)
  void shutdown();

 private:
  MetalContext() = default;
  ~MetalContext();

  // Non-copyable, non-movable (singleton)
  MetalContext(const MetalContext&) = delete;
  MetalContext& operator=(const MetalContext&) = delete;
  MetalContext(MetalContext&&) = delete;
  MetalContext& operator=(MetalContext&&) = delete;

  MetalPtr<MTL::Device> device;
  MetalPtr<MTL::CommandQueue> commandQueue;

  mutable std::mutex mutex;
  bool initialized = false;
};

}  // namespace metal
}  // namespace ts
