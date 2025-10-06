#if defined(RUN_PAMPLEJUCE_TESTS)

#include <ts_metal_accel/ts_metal_accel.h>
#include <ts_metal_accel_fdtd/ts_metal_accel_fdtd.h>

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("FDTD GPU aligns with CPU baseline", "[gpu][fdtd]") {
  using namespace ts::metal;
  using namespace ts::metal::fdtd;

  if (!ts::metal::isMetalAvailable()) {
    WARN("Metal unavailable; skipping GPU FDTD comparison test");
    return;
  }

  constexpr int numSamples = 16;

  FDTDReverbConfig cfg{};
  cfg.solver.grid = {6, 6, 6};
  cfg.solver.dx = 0.03f;
  cfg.solver.dt = 1.0f / 48000.0f;
  cfg.solver.soundSpeed = 343.0f;
  cfg.solver.density = 1.2f;
  cfg.solver.boundaryAttenuation = 0.99f;
  cfg.wetLevel = 1.0f;
  cfg.dryLevel = 0.0f;

  cfg.sources[0] = {2, 2, 2, 1.0f};
  cfg.sources[1] = {3, 2, 2, 1.0f};
  cfg.mics[0] = {3, 3, 3, 1.0f};
  cfg.mics[1] = {4, 3, 3, 1.0f};

  std::string error;
  FDTDReverbEngine engine;
  REQUIRE(engine.prepare(cfg, error));

  if (!engine.usesGPU()) {
    WARN("FDTD engine fell back to CPU; skipping GPU comparison test");
    return;
  }

  engine.reset();

  std::array<float, numSamples> impulseL{};
  std::array<float, numSamples> impulseR{};
  impulseL[0] = 1.0f;
  impulseR[0] = 0.5f;

  const float* inputPtrs[2] = {impulseL.data(), impulseR.data()};
  std::array<float, numSamples> gpuOutLeft{};
  std::array<float, numSamples> gpuOutRight{};
  float* outputPtrs[2] = {gpuOutLeft.data(), gpuOutRight.data()};

  // Process block twice: first block primes async pipeline (returns zeros),
  // second block returns the actual results from first block (1-block latency)
  engine.processBlock(inputPtrs, 2, outputPtrs, 2, numSamples);
  engine.processBlock(inputPtrs, 2, outputPtrs, 2, numSamples);

  // CPU reference
  FDTDCPUSolver cpuSolver{cfg.solver};
  cpuSolver.reset();

  std::array<float, numSamples> cpuOutLeft{};
  std::array<float, numSamples> cpuOutRight{};

  for (int n = 0; n < numSamples; ++n) {
    cpuSolver.addPressureImpulse(cfg.sources[0].x, cfg.sources[0].y,
                                 cfg.sources[0].z,
                                 impulseL[n] * cfg.sources[0].gain);
    cpuSolver.addPressureImpulse(cfg.sources[1].x, cfg.sources[1].y,
                                 cfg.sources[1].z,
                                 impulseR[n] * cfg.sources[1].gain);
    cpuSolver.step();

    cpuOutLeft[n] =
        cpuSolver.pressureAt(cfg.mics[0].x, cfg.mics[0].y, cfg.mics[0].z) *
        cfg.mics[0].gain;
    cpuOutRight[n] =
        cpuSolver.pressureAt(cfg.mics[1].x, cfg.mics[1].y, cfg.mics[1].z) *
        cfg.mics[1].gain;
  }

  for (int n = 0; n < numSamples; ++n) {
    REQUIRE(gpuOutLeft[n] == Catch::Approx(cpuOutLeft[n]).margin(1e-3f));
    REQUIRE(gpuOutRight[n] == Catch::Approx(cpuOutRight[n]).margin(1e-3f));
  }
}

#endif  // RUN_PAMPLEJUCE_TESTS
