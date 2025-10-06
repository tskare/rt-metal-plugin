#include <ts_metal_accel_fdtd/ts_metal_accel_fdtd.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("FDTD CPU solver respects Courant condition", "[fdtd]") {
  using namespace ts::metal::fdtd;

  SolverConfig cfg{};
  cfg.grid = {4, 4, 4};
  cfg.dx = 0.03f;
  cfg.dt = 1.0f / 48000.0f;
  cfg.boundaryAttenuation = 0.95f;

  REQUIRE_NOTHROW(FDTDCPUSolver{cfg});

  cfg.dt = 1.0f / 1000.0f;  // violates Courant
  REQUIRE_THROWS_AS(FDTDCPUSolver{cfg}, std::invalid_argument);
}

TEST_CASE("Impulse energy decays over time", "[fdtd]") {
  using namespace ts::metal::fdtd;

  SolverConfig cfg{};
  cfg.grid = {6, 6, 6};
  cfg.dx = 0.03f;
  cfg.dt = 1.0f / 48000.0f;
  cfg.boundaryAttenuation = 0.9f;

  FDTDCPUSolver solver{cfg};
  solver.reset();

  solver.addPressureImpulse(3, 3, 3, 1.0f);

  const double initialEnergy = solver.totalEnergy();
  REQUIRE(initialEnergy > 0.0);

  double previousEnergy = initialEnergy;

  for (int i = 0; i < 20; ++i) {
    solver.step();
    const double energy = solver.totalEnergy();
    REQUIRE(energy <= previousEnergy + 1e-6);
    previousEnergy = energy;
  }

  REQUIRE(previousEnergy < initialEnergy);
}

TEST_CASE("Impulse remains symmetric around centre", "[fdtd]") {
  using namespace ts::metal::fdtd;

  SolverConfig cfg{};
  cfg.grid = {5, 5, 5};
  cfg.dx = 0.03f;
  cfg.dt = 1.0f / 48000.0f;

  FDTDCPUSolver solver{cfg};
  solver.reset();
  solver.addPressureImpulse(2, 2, 2, 1.0f);

  solver.step();

  const auto centre = solver.pressureAt(2, 2, 2);
  const auto xPlus = solver.pressureAt(3, 2, 2);
  const auto xMinus = solver.pressureAt(1, 2, 2);

  REQUIRE(xPlus == Catch::Approx(xMinus).margin(1e-5f));
  REQUIRE(xPlus < centre);
}
