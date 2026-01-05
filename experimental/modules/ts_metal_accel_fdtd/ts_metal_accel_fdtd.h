/*
 * ts-metal-accel-fdtd - GPU-friendly FDTD reverb helpers
 * Copyright (c) 2025 Travis Skare
 *
 * SPDX-License-Identifier: MIT
 */

/*******************************************************************************
 BEGIN_JUCE_MODULE_DECLARATION

  ID:                 ts_metal_accel_fdtd
  vendor:             tskare
  version:            0.1.0
  name:               FDTD Reverb Helpers (Metal)
  description:        CPU/GPU helpers for finite-difference time-domain reverb experiments.
  website:            https://github.com/travisskare/rt-metal-plugin
  license:            MIT
  minimumCppStandard: 23
  dependencies:       ts_metal_accel, juce_core

 END_JUCE_MODULE_DECLARATION
*******************************************************************************/

#pragma once

#include "Source/FDTDCPU.h"
#include "Source/FDTDTypes.h"
#include "Source/FDTDMetal.h"
#include "Source/FDTDReverbEngine.h"
