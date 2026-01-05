#pragma once

/*******************************************************************************
 BEGIN_JUCE_MODULE_DECLARATION

  ID:                 ts_metal_accel_waveguide
  vendor:             tskare
  version:            0.1.0
  name:               Metal Waveguide Accelerator
  description:        GPU-accelerated digital waveguide synthesis using Metal
  license:            MIT
  dependencies:       juce_audio_basics, ts_metal_accel

 END_JUCE_MODULE_DECLARATION
*******************************************************************************/

#include <juce_audio_basics/juce_audio_basics.h>
#include <ts_metal_accel/ts_metal_accel.h>

#include "Source/WaveguideTypes.h"
#include "Source/WaveguideCSVLoader.h"
#include "Source/WaveguideEngine.h"
