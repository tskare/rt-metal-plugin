#include <ts_metal_accel_fdtd/ts_metal_accel_fdtd.h>

#include <algorithm>
#include <array>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <iomanip>
#include <iostream>

// Helper to print grid info and prepare configs - no dynamic benchmark names
template <int GridX, int GridY, int GridZ, int NumSamples>
void printGridInfo() {
  constexpr int totalCells = GridX * GridY * GridZ;
  constexpr float sampleRate = 48000.0f;
  constexpr float blockDurationMs = (NumSamples / sampleRate) * 1000.0f;

  std::cout << "Grid: " << GridX << "×" << GridY << "×" << GridZ << " ("
            << totalCells << " cells)" << std::endl;
  std::cout << "Block size: " << NumSamples << " samples @ 48kHz" << std::endl;
  std::cout << "Block duration: " << std::fixed << std::setprecision(3)
            << blockDurationMs << " ms (real-time deadline)" << std::endl;
}

template <int GridX, int GridY, int GridZ, int NumSamples>
ts::metal::fdtd::FDTDReverbConfig createConfig() {
  using namespace ts::metal::fdtd;
  constexpr float sampleRate = 48000.0f;

  FDTDReverbConfig cfg{};
  cfg.solver.grid = {GridX, GridY, GridZ};
  cfg.solver.dx = 0.03f;
  cfg.solver.dt = 1.0f / sampleRate;
  cfg.solver.soundSpeed = 343.0f;
  cfg.solver.density = 1.2f;
  cfg.solver.boundaryAttenuation = 0.995f;
  cfg.wetLevel = 1.0f;
  cfg.dryLevel = 0.0f;

  // Place sources and mics in reasonable positions
  cfg.sources[0] = {GridX * 0.2f, GridY * 0.2f, GridZ * 0.2f, 1.0f};
  cfg.sources[1] = {GridX * 0.8f, GridY * 0.2f, GridZ * 0.2f, 1.0f};
  cfg.mics[0] = {GridX * 0.5f, GridY * 0.5f, GridZ * 0.5f, 1.0f};
  cfg.mics[1] = {GridX * 0.5f + 0.5f, GridY * 0.5f, GridZ * 0.5f, 1.0f};

  return cfg;
}

TEST_CASE("FDTD Performance: Small Grid (6×6×6, 216 cells)",
          "[bench][fdtd][small]") {
  using namespace ts::metal::fdtd;

  std::cout << "\n=== Small Grid @ 512 samples ===" << std::endl;
  printGridInfo<6, 6, 6, 512>();

  constexpr int NumSamples = 512;
  auto cfg = createConfig<6, 6, 6, NumSamples>();

  std::array<float, NumSamples> inputL{};
  std::array<float, NumSamples> inputR{};
  inputL[0] = 1.0f;
  inputR[0] = 0.5f;

  const float* inputs[2] = {inputL.data(), inputR.data()};
  std::array<float, NumSamples> gpuOutL{};
  std::array<float, NumSamples> gpuOutR{};
  float* outputs[2] = {gpuOutL.data(), gpuOutR.data()};

  FDTDReverbEngine engine;
  std::string error;
  REQUIRE(engine.prepare(cfg, error));

  BENCHMARK("GPU: Small Grid @ 512 samples") {
    engine.reset();
    std::fill(gpuOutL.begin(), gpuOutL.end(), 0.0f);
    std::fill(gpuOutR.begin(), gpuOutR.end(), 0.0f);
    engine.processBlock(inputs, 2, outputs, 2, NumSamples);
  };

  FDTDCPUSolver cpu{cfg.solver};
  BENCHMARK("CPU: Small Grid @ 512 samples") {
    cpu.reset();
    std::array<float, NumSamples> cpuL{};
    std::array<float, NumSamples> cpuR{};
    for (int n = 0; n < NumSamples; ++n) {
      cpu.addPressureImpulse(cfg.sources[0].x, cfg.sources[0].y,
                             cfg.sources[0].z, inputL[n]);
      cpu.addPressureImpulse(cfg.sources[1].x, cfg.sources[1].y,
                             cfg.sources[1].z, inputR[n]);
      cpu.step();
      cpuL[n] = cpu.pressureAt(cfg.mics[0].x, cfg.mics[0].y, cfg.mics[0].z);
      cpuR[n] = cpu.pressureAt(cfg.mics[1].x, cfg.mics[1].y, cfg.mics[1].z);
    }
  };

  std::cout << std::endl;
}

TEST_CASE("FDTD Performance: Medium Grid (10×10×10, 1000 cells)",
          "[bench][fdtd][medium]") {
  using namespace ts::metal::fdtd;

  std::cout << "\n=== Medium Grid @ 512 samples ===" << std::endl;
  printGridInfo<10, 10, 10, 512>();

  constexpr int NumSamples = 512;
  auto cfg = createConfig<10, 10, 10, NumSamples>();

  std::array<float, NumSamples> inputL{};
  std::array<float, NumSamples> inputR{};
  inputL[0] = 1.0f;
  inputR[0] = 0.5f;

  const float* inputs[2] = {inputL.data(), inputR.data()};
  std::array<float, NumSamples> gpuOutL{};
  std::array<float, NumSamples> gpuOutR{};
  float* outputs[2] = {gpuOutL.data(), gpuOutR.data()};

  FDTDReverbEngine engine;
  std::string error;
  REQUIRE(engine.prepare(cfg, error));

  BENCHMARK("GPU: Medium Grid @ 512 samples") {
    engine.reset();
    std::fill(gpuOutL.begin(), gpuOutL.end(), 0.0f);
    std::fill(gpuOutR.begin(), gpuOutR.end(), 0.0f);
    engine.processBlock(inputs, 2, outputs, 2, NumSamples);
  };

  FDTDCPUSolver cpu{cfg.solver};
  BENCHMARK("CPU: Medium Grid @ 512 samples") {
    cpu.reset();
    std::array<float, NumSamples> cpuL{};
    std::array<float, NumSamples> cpuR{};
    for (int n = 0; n < NumSamples; ++n) {
      cpu.addPressureImpulse(cfg.sources[0].x, cfg.sources[0].y,
                             cfg.sources[0].z, inputL[n]);
      cpu.addPressureImpulse(cfg.sources[1].x, cfg.sources[1].y,
                             cfg.sources[1].z, inputR[n]);
      cpu.step();
      cpuL[n] = cpu.pressureAt(cfg.mics[0].x, cfg.mics[0].y, cfg.mics[0].z);
      cpuR[n] = cpu.pressureAt(cfg.mics[1].x, cfg.mics[1].y, cfg.mics[1].z);
    }
  };

  std::cout << std::endl;
}

TEST_CASE("FDTD Performance: Large Grid (20×20×20, 8000 cells)",
          "[bench][fdtd][large]") {
  using namespace ts::metal::fdtd;

  std::cout << "\n=== Large Grid @ 512 samples ===" << std::endl;
  printGridInfo<20, 20, 20, 512>();

  constexpr int NumSamples = 512;
  auto cfg = createConfig<20, 20, 20, NumSamples>();

  std::array<float, NumSamples> inputL{};
  std::array<float, NumSamples> inputR{};
  inputL[0] = 1.0f;
  inputR[0] = 0.5f;

  const float* inputs[2] = {inputL.data(), inputR.data()};
  std::array<float, NumSamples> gpuOutL{};
  std::array<float, NumSamples> gpuOutR{};
  float* outputs[2] = {gpuOutL.data(), gpuOutR.data()};

  FDTDReverbEngine engine;
  std::string error;
  REQUIRE(engine.prepare(cfg, error));

  BENCHMARK("GPU: Large Grid @ 512 samples") {
    engine.reset();
    std::fill(gpuOutL.begin(), gpuOutL.end(), 0.0f);
    std::fill(gpuOutR.begin(), gpuOutR.end(), 0.0f);
    engine.processBlock(inputs, 2, outputs, 2, NumSamples);
  };

  FDTDCPUSolver cpu{cfg.solver};
  BENCHMARK("CPU: Large Grid @ 512 samples") {
    cpu.reset();
    std::array<float, NumSamples> cpuL{};
    std::array<float, NumSamples> cpuR{};
    for (int n = 0; n < NumSamples; ++n) {
      cpu.addPressureImpulse(cfg.sources[0].x, cfg.sources[0].y,
                             cfg.sources[0].z, inputL[n]);
      cpu.addPressureImpulse(cfg.sources[1].x, cfg.sources[1].y,
                             cfg.sources[1].z, inputR[n]);
      cpu.step();
      cpuL[n] = cpu.pressureAt(cfg.mics[0].x, cfg.mics[0].y, cfg.mics[0].z);
      cpuR[n] = cpu.pressureAt(cfg.mics[1].x, cfg.mics[1].y, cfg.mics[1].z);
    }
  };

  std::cout << std::endl;
}

TEST_CASE("FDTD Performance: Single vs Multi Kernel (Medium Grid)",
          "[bench][fdtd][kernel-comparison]") {
  using namespace ts::metal::fdtd;

  std::cout
      << "\n=== Kernel Dispatch Comparison (10×10×10 grid, 512 samples) ==="
      << std::endl;
  printGridInfo<10, 10, 10, 512>();

  constexpr int NumSamples = 512;
  auto cfg = createConfig<10, 10, 10, NumSamples>();

  std::array<float, NumSamples> inputL{};
  std::array<float, NumSamples> inputR{};
  inputL[0] = 1.0f;
  inputR[0] = 0.5f;

  const float* inputs[2] = {inputL.data(), inputR.data()};
  std::array<float, NumSamples> gpuOutL{};
  std::array<float, NumSamples> gpuOutR{};
  float* outputs[2] = {gpuOutL.data(), gpuOutR.data()};

  // Test with single-threadgroup kernel (default)
  {
    FDTDReverbEngine engine;
    std::string error;
    REQUIRE(engine.prepare(cfg, error));

    BENCHMARK("GPU: Single-Threadgroup Kernel") {
      engine.reset();
      std::fill(gpuOutL.begin(), gpuOutL.end(), 0.0f);
      std::fill(gpuOutR.begin(), gpuOutR.end(), 0.0f);
      engine.processBlock(inputs, 2, outputs, 2, NumSamples);
    };
  }

  // Test with multi-kernel approach (by using very small blocks)
  // Small blocks will show the dispatch overhead more clearly
  {
    FDTDReverbEngine engine;
    std::string error;
    REQUIRE(engine.prepare(cfg, error));

    BENCHMARK("GPU: Multi-Kernel (512 individual samples)") {
      engine.reset();
      std::fill(gpuOutL.begin(), gpuOutL.end(), 0.0f);
      std::fill(gpuOutR.begin(), gpuOutR.end(), 0.0f);
      // Process 512 blocks of 1 sample each to maximize kernel dispatch
      // overhead
      for (int i = 0; i < NumSamples; ++i) {
        const float* singleInput[2] = {&inputL[i], &inputR[i]};
        float* singleOutput[2] = {&gpuOutL[i], &gpuOutR[i]};
        engine.processBlock(singleInput, 2, singleOutput, 2, 1);
      }
    };
  }

  std::cout << std::endl;
}

TEST_CASE("FDTD Performance: Block Size Comparison (Medium Grid)",
          "[bench][fdtd][blocksize]") {
  using namespace ts::metal::fdtd;

  std::cout << "\n=== Block Size Comparison (10×10×10 grid) ===" << std::endl;

  SECTION("64 samples") {
    printGridInfo<10, 10, 10, 64>();

    constexpr int NumSamples = 64;
    auto cfg = createConfig<10, 10, 10, NumSamples>();

    std::array<float, NumSamples> inputL{};
    std::array<float, NumSamples> inputR{};
    inputL[0] = 1.0f;
    inputR[0] = 0.5f;

    const float* inputs[2] = {inputL.data(), inputR.data()};
    std::array<float, NumSamples> gpuOutL{};
    std::array<float, NumSamples> gpuOutR{};
    float* outputs[2] = {gpuOutL.data(), gpuOutR.data()};

    FDTDReverbEngine engine;
    std::string error;
    REQUIRE(engine.prepare(cfg, error));

    BENCHMARK("GPU: Medium @ 64 samples") {
      engine.reset();
      std::fill(gpuOutL.begin(), gpuOutL.end(), 0.0f);
      std::fill(gpuOutR.begin(), gpuOutR.end(), 0.0f);
      engine.processBlock(inputs, 2, outputs, 2, NumSamples);
    };

    FDTDCPUSolver cpu{cfg.solver};
    BENCHMARK("CPU: Medium @ 64 samples") {
      cpu.reset();
      std::array<float, NumSamples> cpuL{};
      std::array<float, NumSamples> cpuR{};
      for (int n = 0; n < NumSamples; ++n) {
        cpu.addPressureImpulse(cfg.sources[0].x, cfg.sources[0].y,
                               cfg.sources[0].z, inputL[n]);
        cpu.addPressureImpulse(cfg.sources[1].x, cfg.sources[1].y,
                               cfg.sources[1].z, inputR[n]);
        cpu.step();
        cpuL[n] = cpu.pressureAt(cfg.mics[0].x, cfg.mics[0].y, cfg.mics[0].z);
        cpuR[n] = cpu.pressureAt(cfg.mics[1].x, cfg.mics[1].y, cfg.mics[1].z);
      }
    };
  }

  SECTION("128 samples") {
    printGridInfo<10, 10, 10, 128>();

    constexpr int NumSamples = 128;
    auto cfg = createConfig<10, 10, 10, NumSamples>();

    std::array<float, NumSamples> inputL{};
    std::array<float, NumSamples> inputR{};
    inputL[0] = 1.0f;
    inputR[0] = 0.5f;

    const float* inputs[2] = {inputL.data(), inputR.data()};
    std::array<float, NumSamples> gpuOutL{};
    std::array<float, NumSamples> gpuOutR{};
    float* outputs[2] = {gpuOutL.data(), gpuOutR.data()};

    FDTDReverbEngine engine;
    std::string error;
    REQUIRE(engine.prepare(cfg, error));

    BENCHMARK("GPU: Medium @ 128 samples") {
      engine.reset();
      std::fill(gpuOutL.begin(), gpuOutL.end(), 0.0f);
      std::fill(gpuOutR.begin(), gpuOutR.end(), 0.0f);
      engine.processBlock(inputs, 2, outputs, 2, NumSamples);
    };

    FDTDCPUSolver cpu{cfg.solver};
    BENCHMARK("CPU: Medium @ 128 samples") {
      cpu.reset();
      std::array<float, NumSamples> cpuL{};
      std::array<float, NumSamples> cpuR{};
      for (int n = 0; n < NumSamples; ++n) {
        cpu.addPressureImpulse(cfg.sources[0].x, cfg.sources[0].y,
                               cfg.sources[0].z, inputL[n]);
        cpu.addPressureImpulse(cfg.sources[1].x, cfg.sources[1].y,
                               cfg.sources[1].z, inputR[n]);
        cpu.step();
        cpuL[n] = cpu.pressureAt(cfg.mics[0].x, cfg.mics[0].y, cfg.mics[0].z);
        cpuR[n] = cpu.pressureAt(cfg.mics[1].x, cfg.mics[1].y, cfg.mics[1].z);
      }
    };
  }

  SECTION("256 samples") {
    printGridInfo<10, 10, 10, 256>();

    constexpr int NumSamples = 256;
    auto cfg = createConfig<10, 10, 10, NumSamples>();

    std::array<float, NumSamples> inputL{};
    std::array<float, NumSamples> inputR{};
    inputL[0] = 1.0f;
    inputR[0] = 0.5f;

    const float* inputs[2] = {inputL.data(), inputR.data()};
    std::array<float, NumSamples> gpuOutL{};
    std::array<float, NumSamples> gpuOutR{};
    float* outputs[2] = {gpuOutL.data(), gpuOutR.data()};

    FDTDReverbEngine engine;
    std::string error;
    REQUIRE(engine.prepare(cfg, error));

    BENCHMARK("GPU: Medium @ 256 samples") {
      engine.reset();
      std::fill(gpuOutL.begin(), gpuOutL.end(), 0.0f);
      std::fill(gpuOutR.begin(), gpuOutR.end(), 0.0f);
      engine.processBlock(inputs, 2, outputs, 2, NumSamples);
    };

    FDTDCPUSolver cpu{cfg.solver};
    BENCHMARK("CPU: Medium @ 256 samples") {
      cpu.reset();
      std::array<float, NumSamples> cpuL{};
      std::array<float, NumSamples> cpuR{};
      for (int n = 0; n < NumSamples; ++n) {
        cpu.addPressureImpulse(cfg.sources[0].x, cfg.sources[0].y,
                               cfg.sources[0].z, inputL[n]);
        cpu.addPressureImpulse(cfg.sources[1].x, cfg.sources[1].y,
                               cfg.sources[1].z, inputR[n]);
        cpu.step();
        cpuL[n] = cpu.pressureAt(cfg.mics[0].x, cfg.mics[0].y, cfg.mics[0].z);
        cpuR[n] = cpu.pressureAt(cfg.mics[1].x, cfg.mics[1].y, cfg.mics[1].z);
      }
    };
  }

  SECTION("512 samples") {
    printGridInfo<10, 10, 10, 512>();

    constexpr int NumSamples = 512;
    auto cfg = createConfig<10, 10, 10, NumSamples>();

    std::array<float, NumSamples> inputL{};
    std::array<float, NumSamples> inputR{};
    inputL[0] = 1.0f;
    inputR[0] = 0.5f;

    const float* inputs[2] = {inputL.data(), inputR.data()};
    std::array<float, NumSamples> gpuOutL{};
    std::array<float, NumSamples> gpuOutR{};
    float* outputs[2] = {gpuOutL.data(), gpuOutR.data()};

    FDTDReverbEngine engine;
    std::string error;
    REQUIRE(engine.prepare(cfg, error));

    BENCHMARK("GPU: Medium @ 512 samples") {
      engine.reset();
      std::fill(gpuOutL.begin(), gpuOutL.end(), 0.0f);
      std::fill(gpuOutR.begin(), gpuOutR.end(), 0.0f);
      engine.processBlock(inputs, 2, outputs, 2, NumSamples);
    };

    FDTDCPUSolver cpu{cfg.solver};
    BENCHMARK("CPU: Medium @ 512 samples") {
      cpu.reset();
      std::array<float, NumSamples> cpuL{};
      std::array<float, NumSamples> cpuR{};
      for (int n = 0; n < NumSamples; ++n) {
        cpu.addPressureImpulse(cfg.sources[0].x, cfg.sources[0].y,
                               cfg.sources[0].z, inputL[n]);
        cpu.addPressureImpulse(cfg.sources[1].x, cfg.sources[1].y,
                               cfg.sources[1].z, inputR[n]);
        cpu.step();
        cpuL[n] = cpu.pressureAt(cfg.mics[0].x, cfg.mics[0].y, cfg.mics[0].z);
        cpuR[n] = cpu.pressureAt(cfg.mics[1].x, cfg.mics[1].y, cfg.mics[1].z);
      }
    };
  }
}
