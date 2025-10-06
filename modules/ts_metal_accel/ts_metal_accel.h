/*
 * ts-metal-accel - GPU-accelerated audio processing with Metal
 * Copyright (c) 2025 Travis Skare
 *
 * SPDX-License-Identifier: MIT
 */

/*******************************************************************************
 The block below describes the properties of this module, and is read by
 the JUCE CMake system to automatically generate build configuration.
 For details about the syntax, see the JUCE Module Format.md documentation.


 BEGIN_JUCE_MODULE_DECLARATION

  ID:                 ts_metal_accel
  vendor:             tskare
  version:            0.1.0
  name:               GPU-Accelerated Audio Processing (Metal)
  description:        JUCE module for GPU-accelerated audio processing using Apple Metal.
                      Provides high-performance compute kernels for audio effects and synthesis.
                      Phase 1: Synchronous/blocking implementation for validation.
  website:            https://github.com/travisskare/rt-metal-plugin
  license:            MIT
  minimumCppStandard: 23
  dependencies:       juce_core, juce_audio_basics
  searchpaths:        ts_metal_accel/third_party/metal-cpp

  OSXFrameworks:      Metal Foundation QuartzCore

 END_JUCE_MODULE_DECLARATION

*******************************************************************************/

#pragma once
#define TS_METAL_ACCEL_H_INCLUDED

//==============================================================================
// Module configuration

/** Config: TS_METAL_ACCEL_ENABLE_ASSERTIONS
    Enable runtime assertions for debugging Metal operations.
    Enabled by default in debug builds.
*/
#ifndef TS_METAL_ACCEL_ENABLE_ASSERTIONS
 #if JUCE_DEBUG
  #define TS_METAL_ACCEL_ENABLE_ASSERTIONS 1
 #else
  #define TS_METAL_ACCEL_ENABLE_ASSERTIONS 0
 #endif
#endif

//==============================================================================
// Include module headers

#include "Source/MetalUtils.h"
#include "Source/MetalContext.h"
#include "Source/MetalAccelerator.h"

//==============================================================================
// Version information

namespace ts {
namespace metal {

constexpr int versionMajor = 0;
constexpr int versionMinor = 1;
constexpr int versionPatch = 0;

/**
 * Get the module version as a string.
 */
inline std::string getVersionString()
{
    return std::to_string(versionMajor) + "." +
           std::to_string(versionMinor) + "." +
           std::to_string(versionPatch);
}

/**
 * Check if Metal is available on this system.
 */
inline bool isMetalAvailable()
{
    auto& ctx = MetalContext::getInstance();
    auto result = ctx.initialize();
    return result.has_value() && ctx.isInitialized();
}

} // namespace metal
} // namespace ts
