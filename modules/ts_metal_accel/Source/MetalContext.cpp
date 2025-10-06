/*
 * ts-metal-accel - GPU-accelerated audio processing with Metal
 * Copyright (c) 2025 Travis Skare
 *
 * SPDX-License-Identifier: MIT
 */

#include "MetalContext.h"

namespace ts {
namespace metal {

MetalContext& MetalContext::getInstance() {
  static MetalContext instance;
  return instance;
}

std::expected<void, MetalErrorInfo> MetalContext::initialize() {
  std::lock_guard<std::mutex> lock(mutex);

  if (initialized) return {};

  // Create the default system device
  MTL::Device* dev = MTL::CreateSystemDefaultDevice();
  if (!dev) {
    return std::unexpected(
        MetalErrorInfo{MetalError::DeviceNotFound,
                       "Failed to create default Metal device. Metal may not "
                       "be available on this system."});
  }

  device.reset(dev);

  // Create a command queue for submitting work
  MTL::CommandQueue* queue = device->newCommandQueue();
  if (!queue) {
    device.reset();
    return std::unexpected(MetalErrorInfo{
        MetalError::DeviceNotFound, "Failed to create Metal command queue."});
  }

  commandQueue.reset(queue);
  initialized = true;
  return {};
}

void MetalContext::shutdown() {
  std::lock_guard<std::mutex> lock(mutex);

  // Release in reverse order of creation
  commandQueue.reset();
  device.reset();
  initialized = false;
}

MetalContext::~MetalContext() { shutdown(); }

}  // namespace metal
}  // namespace ts
