#pragma once

#include <string>

#include "FDTDMetalKernels.h"

namespace ts::metal::fdtd {

inline std::string getMetalKernelSource() {
  return std::string{kFDTDMetalKernels};
}

}  // namespace ts::metal::fdtd
