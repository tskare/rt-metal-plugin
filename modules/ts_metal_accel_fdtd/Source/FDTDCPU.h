#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ts::metal::fdtd {

struct GridSize {
  int nx = 0;
  int ny = 0;
  int nz = 0;
};

struct SolverConfig {
  GridSize grid;
  float dx = 0.03f;            // spatial step in metres
  float dt = 1.0f / 48000.0f;  // time step in seconds
  float soundSpeed = 343.0f;   // metres per second
  float density = 1.2f;        // kg / m^3
  float boundaryAttenuation =
      0.98f;  // multiply velocity/pressure at boundaries
};

class FDTDCPUSolver {
 public:
  FDTDCPUSolver() = default;
  explicit FDTDCPUSolver(const SolverConfig& cfg);

  void configure(const SolverConfig& cfg);
  void reset();

  void addPressureImpulse(int x, int y, int z, float amplitude);
  void step();

  [[nodiscard]] float pressureAt(int x, int y, int z) const;
  [[nodiscard]] float pressureAtInterpolated(float x, float y, float z) const;
  [[nodiscard]] double totalEnergy() const;
  [[nodiscard]] const SolverConfig& getConfig() const { return config; }

 private:
  [[nodiscard]] std::size_t index(int x, int y, int z) const;
  void allocateBuffers();

  SolverConfig config{};
  std::vector<float> pressure[2];
  std::vector<float> velocityX[2];
  std::vector<float> velocityY[2];
  std::vector<float> velocityZ[2];
  int activeBuffer = 0;

  float coeffVelocity = 0.0f;
  float coeffPressure = 0.0f;
};

}  // namespace ts::metal::fdtd
