#include <ts_metal_accel_fdtd/ts_metal_accel_fdtd.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("FDTD trilinear interpolation for microphones",
          "[fdtd][interpolation]") {
  using namespace ts::metal::fdtd;

  SECTION("CPU solver interpolation") {
    SolverConfig cfg{};
    cfg.grid = {5, 5, 5};
    cfg.dx = 0.03f;
    cfg.dt = 1.0f / 48000.0f;
    cfg.boundaryAttenuation = 1.0f;  // No attenuation for testing

    FDTDCPUSolver solver{cfg};
    solver.reset();

    // Set up a unit cube with known values at all 8 corners (2,2,2) to (3,3,3)
    // This allows us to test interpolation at (2.5, 2.5, 2.5) - the center of
    // the cube
    solver.addPressureImpulse(2, 2, 2, 1.0f);
    solver.addPressureImpulse(3, 2, 2, 2.0f);
    solver.addPressureImpulse(2, 3, 2, 3.0f);
    solver.addPressureImpulse(3, 3, 2, 4.0f);
    solver.addPressureImpulse(2, 2, 3, 5.0f);
    solver.addPressureImpulse(3, 2, 3, 6.0f);
    solver.addPressureImpulse(2, 3, 3, 7.0f);
    solver.addPressureImpulse(3, 3, 3, 8.0f);

    // Test interpolation at the center of the unit cube (should be average of
    // all 8 corners)
    const float centerValue = solver.pressureAtInterpolated(2.5f, 2.5f, 2.5f);
    const float expectedCenter =
        (1.0f + 2.0f + 3.0f + 4.0f + 5.0f + 6.0f + 7.0f + 8.0f) / 8.0f;
    REQUIRE(centerValue == Catch::Approx(expectedCenter).margin(0.01f));

    // Test interpolation at exact grid points (should match exact values)
    REQUIRE(solver.pressureAtInterpolated(2.0f, 2.0f, 2.0f) ==
            Catch::Approx(1.0f).margin(0.001f));
    REQUIRE(solver.pressureAtInterpolated(3.0f, 3.0f, 3.0f) ==
            Catch::Approx(8.0f).margin(0.001f));

    // Test interpolation at fractional positions along X axis
    const float midX = solver.pressureAtInterpolated(2.5f, 2.0f, 2.0f);
    const float expectedMidX = (1.0f + 2.0f) / 2.0f;
    REQUIRE(midX == Catch::Approx(expectedMidX).margin(0.01f));
  }

  SECTION("FDTD engine with fractional microphone positions") {
    FDTDReverbConfig cfg{};
    cfg.solver.grid = {8, 8, 8};
    cfg.solver.dx = 0.03f;
    cfg.solver.dt = 1.0f / 48000.0f;
    cfg.solver.boundaryAttenuation = 0.99f;
    cfg.wetLevel = 1.0f;
    cfg.dryLevel = 0.0f;

    // Place sources at integer positions
    cfg.sources[0] = {2.0f, 2.0f, 2.0f, 1.0f};
    cfg.sources[1] = {5.0f, 2.0f, 2.0f, 1.0f};

    // Place microphones at fractional positions to test interpolation
    cfg.mics[0] = {3.5f, 3.5f, 3.5f, 1.0f};     // Between grid points
    cfg.mics[1] = {4.25f, 4.75f, 3.33f, 1.0f};  // Arbitrary fractional position

    FDTDReverbEngine engine;
    std::string error;
    REQUIRE(engine.prepare(cfg, error));
    REQUIRE(error.empty());

    engine.reset();

    // Process a single impulse
    constexpr int numSamples = 32;
    float inputL[numSamples] = {};
    float inputR[numSamples] = {};
    inputL[0] = 1.0f;  // Impulse in left channel

    const float* inputs[2] = {inputL, inputR};
    float outputL[numSamples] = {};
    float outputR[numSamples] = {};
    float* outputs[2] = {outputL, outputR};

    engine.processBlock(inputs, 2, outputs, 2, numSamples);

    // Verify that we get non-zero output (wave has propagated)
    bool hasOutput = false;
    for (int i = 0; i < numSamples; ++i) {
      if (std::abs(outputL[i]) > 0.0001f || std::abs(outputR[i]) > 0.0001f) {
        hasOutput = true;
        break;
      }
    }
    REQUIRE(hasOutput);

    // Compare integer vs fractional mic positions
    // Reset and place mics at integer positions
    FDTDReverbConfig cfg2 = cfg;
    cfg2.mics[0] = {3.0f, 3.0f, 3.0f, 1.0f};  // Nearest integer to 3.5
    cfg2.mics[1] = {4.0f, 5.0f, 3.0f, 1.0f};  // Nearest integers

    FDTDReverbEngine engine2;
    REQUIRE(engine2.prepare(cfg2, error));
    engine2.reset();

    float outputL2[numSamples] = {};
    float outputR2[numSamples] = {};
    float* outputs2[2] = {outputL2, outputR2};

    engine2.processBlock(inputs, 2, outputs2, 2, numSamples);

    // The interpolated version should produce different (smoother) output
    float diff = 0.0f;
    for (int i = 0; i < numSamples; ++i) {
      diff += std::abs(outputL[i] - outputL2[i]);
      diff += std::abs(outputR[i] - outputR2[i]);
    }

    // There should be some difference due to interpolation
    REQUIRE(diff > 0.0001f);
  }
}