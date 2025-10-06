#pragma once

#include <cstdint>
#include <vector>

namespace ts::metal::fdtd {

struct GridDimensions {
  std::uint32_t nx = 0;
  std::uint32_t ny = 0;
  std::uint32_t nz = 0;

  [[nodiscard]] std::uint32_t cellCount() const noexcept {
    return nx * ny * nz;
  }
};

struct Uniforms {
  GridDimensions grid{};
  float coeffVelocity = 0.0f;
  float coeffPressure = 0.0f;
  float boundaryAttenuation = 0.0f;
};

struct SourceCommand {
  std::uint32_t index = 0;
};

struct MicCommand {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float gain = 1.0f;
};

using SourceCommandList = std::vector<SourceCommand>;
using MicCommandList = std::vector<MicCommand>;

}  // namespace ts::metal::fdtd
