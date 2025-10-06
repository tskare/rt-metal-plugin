/*
 * ts-metal-accel - GPU-accelerated audio processing with Metal
 * Copyright (c) 2025 Travis Skare
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <array>
#include <atomic>
#include <cassert>
#include <expected>
#include <string>

namespace ts {
namespace metal {

// Error types for std::expected usage
enum class MetalError {
  DeviceNotFound,
  CompilationFailed,
  InvalidKernel,
  InvalidBuffer,
  ExecutionFailed,
  NotInitialized,
  AlreadyInitialized,
  Unknown
};

struct MetalErrorInfo {
  MetalError code;
  std::string message;

  MetalErrorInfo(MetalError c, std::string msg = "")
      : code(c), message(std::move(msg)) {}
};

/**
 * RAII smart pointer for Metal objects using Manual Retain-Release (MRR).
 *
 * Bridges the gap between metal-cpp's manual memory management and modern C++
 * RAII. Automatically manages retain/release calls to prevent leaks and
 * use-after-free bugs.
 *
 * Usage:
 *   MetalPtr<MTL::Device> device{MTL::CreateSystemDefaultDevice()};
 *   // device->methodCall()
 *   // Automatically released when MetalPtr goes out of scope
 */
template <typename T>
class MetalPtr {
 public:
  MetalPtr() : obj(nullptr) {}

  // Takes ownership of a Metal object (assumes already retained)
  explicit MetalPtr(T* p) : obj(p) {}

  ~MetalPtr() {
    if (obj) obj->release();
  }

  // Copy semantics: retain the underlying object
  MetalPtr(const MetalPtr& other) : obj(other.obj) {
    if (obj) obj->retain();
  }

  MetalPtr& operator=(const MetalPtr& other) {
    if (this != &other) {
      if (obj) obj->release();
      obj = other.obj;
      if (obj) obj->retain();
    }
    return *this;
  }

  // Move semantics: transfer ownership without retain/release
  MetalPtr(MetalPtr&& other) noexcept : obj(other.obj) { other.obj = nullptr; }

  MetalPtr& operator=(MetalPtr&& other) noexcept {
    if (this != &other) {
      if (obj) obj->release();
      obj = other.obj;
      other.obj = nullptr;
    }
    return *this;
  }

  // Raw pointer access
  T* get() const { return obj; }
  T* operator->() const {
    assert(obj && "Dereferencing null MetalPtr");
    return obj;
  }

  explicit operator bool() const { return obj != nullptr; }

  // Release ownership and return raw pointer (caller must manage)
  T* release() {
    T* temp = obj;
    obj = nullptr;
    return temp;
  }

  // Reset to a new object (releases old one)
  void reset(T* p = nullptr) {
    if (obj) obj->release();
    obj = p;
  }

 private:
  T* obj;
};

/**
 * Helper to create a MetalPtr from a newly created Metal object.
 * For use with methods that return new/alloc/copy/mutableCopy (already
 * retained).
 */
template <typename T>
MetalPtr<T> makeMetalPtr(T* ptr) {
  return MetalPtr<T>(ptr);
}

/**
 * Helper to retain and wrap an existing Metal object.
 * For use with getter methods that return unretained pointers.
 */
template <typename T>
MetalPtr<T> retainMetalPtr(T* ptr) {
  if (ptr) ptr->retain();
  return MetalPtr<T>(ptr);
}

/**
 * Scoped autorelease pool helper for metal-cpp objects on non-ObjC threads.
 */
class AutoreleasePool {
 public:
  AutoreleasePool() : pool(NS::AutoreleasePool::alloc()->init()) {}

  ~AutoreleasePool() {
    if (pool) pool->release();
  }

 private:
  NS::AutoreleasePool* pool = nullptr;
};

/**
 * Lock-free single-producer/single-consumer ring buffer.
 * Capacity must be a power of two. One slot is reserved to
 * distinguish between empty and full states.
 */
template <typename T, size_t Capacity>
class SpscQueue {
  static_assert((Capacity > 1) && ((Capacity & (Capacity - 1)) == 0),
                "Capacity must be a power of two and > 1");

 public:
  bool push(const T& value) {
    auto currentHead = head.load(std::memory_order_relaxed);
    auto nextHead = increment(currentHead);
    if (nextHead == tail.load(std::memory_order_acquire))
      return false;  // queue full

    buffer[currentHead] = value;
    head.store(nextHead, std::memory_order_release);
    return true;
  }

  bool pop(T& value) {
    auto currentTail = tail.load(std::memory_order_relaxed);
    if (currentTail == head.load(std::memory_order_acquire))
      return false;  // queue empty

    value = buffer[currentTail];
    tail.store(increment(currentTail), std::memory_order_release);
    return true;
  }

  bool empty() const {
    return tail.load(std::memory_order_acquire) ==
           head.load(std::memory_order_acquire);
  }

  void reset() {
    head.store(0, std::memory_order_relaxed);
    tail.store(0, std::memory_order_relaxed);
  }

 private:
  static constexpr size_t increment(size_t index) {
    return (index + 1) & (Capacity - 1);
  }

  std::array<T, Capacity> buffer{};
  std::atomic<size_t> head{0};
  std::atomic<size_t> tail{0};
};

}  // namespace metal
}  // namespace ts
