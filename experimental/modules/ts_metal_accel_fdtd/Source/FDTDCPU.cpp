#include "FDTDCPU.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ts::metal::fdtd {
namespace {
constexpr float minDimension = 2.0f;

inline bool isBoundary(int coord, int extent) {
  return coord == 0 || coord == extent - 1;
}
}  // namespace

FDTDCPUSolver::FDTDCPUSolver(const SolverConfig& cfg) { configure(cfg); }

void FDTDCPUSolver::configure(const SolverConfig& cfg) {
  if (cfg.grid.nx < 2 || cfg.grid.ny < 2 || cfg.grid.nz < 2)
    throw std::invalid_argument(
        "Grid dimensions must be at least 2 in each direction");

  if (cfg.dx <= 0.0f || cfg.dt <= 0.0f)
    throw std::invalid_argument("Spatial and temporal steps must be positive");

  const float courant = cfg.soundSpeed * cfg.dt / cfg.dx;
  if (courant > 1.0f / std::sqrt(3.0f))
    throw std::invalid_argument(
        "Courant condition violated: reduce dt or increase dx");

  config = cfg;
  allocateBuffers();
  reset();

  coeffVelocity = cfg.dt / (cfg.density * cfg.dx);
  coeffPressure =
      cfg.density * cfg.soundSpeed * cfg.soundSpeed * cfg.dt / cfg.dx;
}

void FDTDCPUSolver::allocateBuffers() {
  const std::size_t total = static_cast<std::size_t>(config.grid.nx) *
                            static_cast<std::size_t>(config.grid.ny) *
                            static_cast<std::size_t>(config.grid.nz);

  for (auto* buffer :
       {&pressure[0], &pressure[1], &velocityX[0], &velocityX[1], &velocityY[0],
        &velocityY[1], &velocityZ[0], &velocityZ[1]})
    buffer->assign(total, 0.0f);

  activeBuffer = 0;
}

void FDTDCPUSolver::reset() {
  for (auto* buffer :
       {&pressure[0], &pressure[1], &velocityX[0], &velocityX[1], &velocityY[0],
        &velocityY[1], &velocityZ[0], &velocityZ[1]})
    std::fill(buffer->begin(), buffer->end(), 0.0f);

  activeBuffer = 0;
}

std::size_t FDTDCPUSolver::index(int x, int y, int z) const {
  return static_cast<std::size_t>(x) +
         static_cast<std::size_t>(config.grid.nx) *
             (static_cast<std::size_t>(y) +
              static_cast<std::size_t>(config.grid.ny) *
                  static_cast<std::size_t>(z));
}

void FDTDCPUSolver::addPressureImpulse(int x, int y, int z, float amplitude) {
  const auto idx = index(x, y, z);
  pressure[activeBuffer][idx] += amplitude;
}

void FDTDCPUSolver::step() {
  const int nx = config.grid.nx;
  const int ny = config.grid.ny;
  const int nz = config.grid.nz;

  const int writeIndex = 1 - activeBuffer;

  const auto& pCurr = pressure[activeBuffer];
  const auto& vxCurr = velocityX[activeBuffer];
  const auto& vyCurr = velocityY[activeBuffer];
  const auto& vzCurr = velocityZ[activeBuffer];

  auto& vxNext = velocityX[writeIndex];
  auto& vyNext = velocityY[writeIndex];
  auto& vzNext = velocityZ[writeIndex];
  auto& pNext = pressure[writeIndex];

  const float boundaryAttenuation = config.boundaryAttenuation;

  for (int z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        const std::size_t idx = index(x, y, z);

        const float px =
            (x < nx - 1) ? (pCurr[index(x + 1, y, z)] - pCurr[idx]) : 0.0f;
        const float py =
            (y < ny - 1) ? (pCurr[index(x, y + 1, z)] - pCurr[idx]) : 0.0f;
        const float pz =
            (z < nz - 1) ? (pCurr[index(x, y, z + 1)] - pCurr[idx]) : 0.0f;

        auto dampingX = (isBoundary(x, nx) ? boundaryAttenuation : 1.0f);
        auto dampingY = (isBoundary(y, ny) ? boundaryAttenuation : 1.0f);
        auto dampingZ = (isBoundary(z, nz) ? boundaryAttenuation : 1.0f);

        vxNext[idx] = dampingX * (vxCurr[idx] - coeffVelocity * px);
        vyNext[idx] = dampingY * (vyCurr[idx] - coeffVelocity * py);
        vzNext[idx] = dampingZ * (vzCurr[idx] - coeffVelocity * pz);
      }
    }
  }

  for (int z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        const std::size_t idx = index(x, y, z);

        float dvx = 0.0f;
        float dvy = 0.0f;
        float dvz = 0.0f;

        if (x > 0)
          dvx += vxNext[idx] - vxNext[index(x - 1, y, z)];
        else
          dvx += vxNext[idx];

        if (y > 0)
          dvy += vyNext[idx] - vyNext[index(x, y - 1, z)];
        else
          dvy += vyNext[idx];

        if (z > 0)
          dvz += vzNext[idx] - vzNext[index(x, y, z - 1)];
        else
          dvz += vzNext[idx];

        float divergence = dvx + dvy + dvz;

        const float damping =
            (isBoundary(x, nx) || isBoundary(y, ny) || isBoundary(z, nz))
                ? boundaryAttenuation
                : 1.0f;

        pNext[idx] = damping * (pCurr[idx] - coeffPressure * divergence);
      }
    }
  }

  activeBuffer = writeIndex;
}

float FDTDCPUSolver::pressureAt(int x, int y, int z) const {
  return pressure[activeBuffer][index(x, y, z)];
}

float FDTDCPUSolver::pressureAtInterpolated(float x, float y, float z) const {
  // Clamp to valid grid range
  x = std::max(0.0f, std::min(x, static_cast<float>(config.grid.nx - 1)));
  y = std::max(0.0f, std::min(y, static_cast<float>(config.grid.ny - 1)));
  z = std::max(0.0f, std::min(z, static_cast<float>(config.grid.nz - 1)));

  // Get integer parts (lower corner of the interpolation cube)
  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int z0 = static_cast<int>(std::floor(z));

  // Get fractional parts for interpolation weights
  const float fx = x - static_cast<float>(x0);
  const float fy = y - static_cast<float>(y0);
  const float fz = z - static_cast<float>(z0);

  // Ensure we don't go out of bounds for the upper corner
  const int x1 = std::min(x0 + 1, config.grid.nx - 1);
  const int y1 = std::min(y0 + 1, config.grid.ny - 1);
  const int z1 = std::min(z0 + 1, config.grid.nz - 1);

  // Get the 8 corner values of the interpolation cube
  const float p000 = pressure[activeBuffer][index(x0, y0, z0)];
  const float p001 = pressure[activeBuffer][index(x0, y0, z1)];
  const float p010 = pressure[activeBuffer][index(x0, y1, z0)];
  const float p011 = pressure[activeBuffer][index(x0, y1, z1)];
  const float p100 = pressure[activeBuffer][index(x1, y0, z0)];
  const float p101 = pressure[activeBuffer][index(x1, y0, z1)];
  const float p110 = pressure[activeBuffer][index(x1, y1, z0)];
  const float p111 = pressure[activeBuffer][index(x1, y1, z1)];

  // Trilinear interpolation
  // Interpolate along x first
  const float px00 = p000 * (1.0f - fx) + p100 * fx;
  const float px01 = p001 * (1.0f - fx) + p101 * fx;
  const float px10 = p010 * (1.0f - fx) + p110 * fx;
  const float px11 = p011 * (1.0f - fx) + p111 * fx;

  // Then interpolate along y
  const float pxy0 = px00 * (1.0f - fy) + px10 * fy;
  const float pxy1 = px01 * (1.0f - fy) + px11 * fy;

  // Finally interpolate along z
  return pxy0 * (1.0f - fz) + pxy1 * fz;
}

double FDTDCPUSolver::totalEnergy() const {
  const auto& p = pressure[activeBuffer];
  const auto& vx = velocityX[activeBuffer];
  const auto& vy = velocityY[activeBuffer];
  const auto& vz = velocityZ[activeBuffer];

  double sum = 0.0;
  for (std::size_t i = 0; i < p.size(); ++i) {
    sum += static_cast<double>(p[i]) * static_cast<double>(p[i]);
    sum += static_cast<double>(vx[i]) * static_cast<double>(vx[i]);
    sum += static_cast<double>(vy[i]) * static_cast<double>(vy[i]);
    sum += static_cast<double>(vz[i]) * static_cast<double>(vz[i]);
  }

  return sum;
}

}  // namespace ts::metal::fdtd
