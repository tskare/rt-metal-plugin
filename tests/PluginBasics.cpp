#include <PluginProcessor.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "helpers/test_helpers.h"

TEST_CASE("one is equal to one", "[dummy]") { REQUIRE(1 == 1); }

TEST_CASE("Plugin instance", "[instance]") {
  PluginProcessor testPlugin;

  SECTION("name") {
    CHECK_THAT(
        testPlugin.getName().toStdString(),
        Catch::Matchers::Equals(juce::String(JucePlugin_Name).toStdString()));
  }
}

#ifdef PAMPLEJUCE_IPP
#include <ipp.h>

TEST_CASE("IPP version", "[ipp]") {
  CHECK_THAT(ippsGetLibVersion()->Version,
             Catch::Matchers::Equals("2022.2.0 (r0x42db1a66)"));
}
#endif
