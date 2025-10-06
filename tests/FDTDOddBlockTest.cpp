#if defined(RUN_PAMPLEJUCE_TESTS)

#include <ts_metal_accel/ts_metal_accel.h>
#include <ts_metal_accel_fdtd/ts_metal_accel_fdtd.h>

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("FDTD GPU handles consecutive odd-sized blocks", "[gpu][fdtd]") {
  using namespace ts::metal;
  using namespace ts::metal::fdtd;

  if (!ts::metal::isMetalAvailable()) {
    WARN("Metal unavailable; skipping GPU odd block test");
    return;
  }

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
  cfg.mics[0] = {3, 3, 3, 1.0f};

  std::string error;
  FDTDReverbEngine gpuEngine;
  REQUIRE(gpuEngine.prepare(cfg, error));

  if (!gpuEngine.usesGPU()) {
    WARN("FDTD engine fell back to CPU; skipping GPU odd block test");
    return;
  }

  gpuEngine.reset();

  // CPU reference
  FDTDCPUSolver cpuSolver{cfg.solver};
  cpuSolver.reset();

  // With async processing, GPU output from block N contains results from block
  // N-1 So we need to compute and save CPU results, then compare with GPU
  // results from next block

  // Process several consecutive odd-sized blocks to test buffer parity handling
  const int blockSizes[] = {3, 1, 5, 3, 1};  // All odd sizes
  constexpr int numBlocks = 5;
  constexpr int maxBlockSize = 5;

  // Storage for CPU outputs (computed in block N-1, compared against GPU in
  // block N)
  std::array<std::array<float, maxBlockSize>, numBlocks + 1> cpuOutputs{};

  // Block -1: Initial impulse - compute CPU reference for this block
  {
    cpuSolver.addPressureImpulse(cfg.sources[0].x, cfg.sources[0].y,
                                 cfg.sources[0].z, 1.0f * cfg.sources[0].gain);
    for (int n = 0; n < blockSizes[0]; ++n)  // First test block size
    {
      cpuSolver.step();
      cpuOutputs[0][n] =
          cpuSolver.pressureAt(cfg.mics[0].x, cfg.mics[0].y, cfg.mics[0].z) *
          cfg.mics[0].gain;
    }
  }

  // Block 0: Prime GPU async pipeline with impulse (returns zeros)
  {
    std::array<float, maxBlockSize> impulse{};
    impulse[0] = 1.0f;
    const float* inputPtrs[1] = {impulse.data()};
    std::array<float, maxBlockSize> dummyOutput{};
    float* outputPtrs[1] = {dummyOutput.data()};
    gpuEngine.processBlock(inputPtrs, 1, outputPtrs, 1, blockSizes[0]);
  }

  // Blocks 1 to numBlocks: Process with silence, compare against CPU from
  // previous block
  for (int blockIdx = 0; blockIdx < numBlocks; ++blockIdx) {
    const int currentBlockSize = blockSizes[blockIdx];

    // Compute CPU reference for current block (advance by currentBlockSize)
    for (int n = 0; n < currentBlockSize; ++n) {
      cpuSolver.step();
      // Save to next slot if not last iteration
      if (blockIdx < numBlocks - 1) {
        // We'll use this in the NEXT iteration
        cpuOutputs[blockIdx + 1][n] =
            cpuSolver.pressureAt(cfg.mics[0].x, cfg.mics[0].y, cfg.mics[0].z) *
            cfg.mics[0].gain;
      }
    }

    // GPU processing with silence (returns results from PREVIOUS block)
    std::array<float, maxBlockSize> silence{};
    const float* inputPtrs[1] = {silence.data()};
    std::array<float, maxBlockSize> gpuOutput{};
    float* outputPtrs[1] = {gpuOutput.data()};
    gpuEngine.processBlock(inputPtrs, 1, outputPtrs, 1, currentBlockSize);

    // GPU returns results from PREVIOUS submitted block
    // Block 0 returns from priming (blockSizes[0]), Block 1 returns from block
    // 0 (blockSizes[0]), Block 2 returns from block 1 (blockSizes[1]), etc.
    const int returnedBlockSize =
        (blockIdx == 0) ? blockSizes[0] : blockSizes[blockIdx - 1];
    for (int n = 0; n < returnedBlockSize; ++n) {
      INFO("Block " << blockIdx << " (processing " << currentBlockSize
                    << " samples), sample " << n << " (returned from prev)");
      REQUIRE(gpuOutput[n] ==
              Catch::Approx(cpuOutputs[blockIdx][n]).margin(1e-3f));
    }
  }
}

#endif  // RUN_PAMPLEJUCE_TESTS
