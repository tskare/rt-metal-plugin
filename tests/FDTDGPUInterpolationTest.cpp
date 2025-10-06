#if defined(RUN_PAMPLEJUCE_TESTS)

#include <ts_metal_accel/ts_metal_accel.h>
#include <ts_metal_accel_fdtd/ts_metal_accel_fdtd.h>

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

TEST_CASE("FDTD GPU trilinear interpolation matches CPU",
          "[gpu][fdtd][interpolation]") {
  using namespace ts::metal;
  using namespace ts::metal::fdtd;

  if (!ts::metal::isMetalAvailable()) {
    WARN("Metal unavailable; skipping GPU interpolation test");
    return;
  }

  FDTDReverbConfig cfg{};
  cfg.solver.grid = {8, 8, 8};
  cfg.solver.dx = 0.03f;
  cfg.solver.dt = 1.0f / 48000.0f;
  cfg.solver.soundSpeed = 343.0f;
  cfg.solver.density = 1.2f;
  cfg.solver.boundaryAttenuation = 0.99f;
  cfg.wetLevel = 1.0f;
  cfg.dryLevel = 0.0f;

  // Sources at integer positions
  cfg.sources[0] = {2.0f, 2.0f, 2.0f, 1.0f};
  cfg.sources[1] = {5.0f, 2.0f, 2.0f, 0.5f};

  // Mics at FRACTIONAL positions to test interpolation
  cfg.mics[0] = {3.5f, 3.5f, 3.5f, 1.0f};     // Center between cells
  cfg.mics[1] = {4.25f, 4.75f, 3.33f, 0.8f};  // Arbitrary fractional

  std::string error;
  FDTDReverbEngine gpuEngine;
  REQUIRE(gpuEngine.prepare(cfg, error));

  if (!gpuEngine.usesGPU()) {
    WARN("FDTD engine fell back to CPU; skipping GPU interpolation test");
    return;
  }

  gpuEngine.reset();

  // Process impulse
  constexpr int numSamples = 16;
  std::array<float, numSamples> impulseL{};
  std::array<float, numSamples> impulseR{};
  impulseL[0] = 1.0f;
  impulseR[0] = 0.5f;

  const float* inputPtrs[2] = {impulseL.data(), impulseR.data()};
  std::array<float, numSamples> gpuOutL{};
  std::array<float, numSamples> gpuOutR{};
  float* outputPtrs[2] = {gpuOutL.data(), gpuOutR.data()};

  // Process block twice: first block primes async pipeline (returns zeros),
  // second block returns the actual results from first block (1-block latency)
  gpuEngine.processBlock(inputPtrs, 2, outputPtrs, 2, numSamples);
  gpuEngine.processBlock(inputPtrs, 2, outputPtrs, 2, numSamples);

  // CPU reference with SAME fractional positions
  FDTDReverbEngine cpuEngine;
  REQUIRE(cpuEngine.prepare(cfg, error));
  cpuEngine.reset();

  std::array<float, numSamples> cpuOutL{};
  std::array<float, numSamples> cpuOutR{};
  float* cpuOutputPtrs[2] = {cpuOutL.data(), cpuOutR.data()};

  // CPU path doesn't have async latency, but process twice for consistency
  cpuEngine.processBlock(inputPtrs, 2, cpuOutputPtrs, 2, numSamples);
  cpuEngine.processBlock(inputPtrs, 2, cpuOutputPtrs, 2, numSamples);

  // GPU should match CPU (with trilinear interpolation)
  for (int n = 0; n < numSamples; ++n) {
    INFO("Sample " << n);
    REQUIRE(gpuOutL[n] == Catch::Approx(cpuOutL[n]).margin(1e-3f));
    REQUIRE(gpuOutR[n] == Catch::Approx(cpuOutR[n]).margin(1e-3f));
  }

  // Verify we actually got non-trivial output (interpolation happened)
  float sumL = 0.0f, sumR = 0.0f;
  for (int n = 0; n < numSamples; ++n) {
    sumL += std::abs(gpuOutL[n]);
    sumR += std::abs(gpuOutR[n]);
  }
  REQUIRE(sumL > 0.001f);
  REQUIRE(sumR > 0.001f);
}

#endif  // RUN_PAMPLEJUCE_TESTS
