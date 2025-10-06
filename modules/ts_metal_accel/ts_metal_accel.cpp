/*
 * ts-metal-accel - GPU-accelerated audio processing with Metal
 * Copyright (c) 2025 Travis Skare
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * This file generates the implementation for metal-cpp.
 * These macros MUST be defined in exactly one translation unit.
 */

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

// Include module implementation files
#include "Source/MetalContext.cpp"
#include "Source/MetalAccelerator.cpp"
