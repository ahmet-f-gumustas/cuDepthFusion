#include <gtest/gtest.h>

#include <functional>
#include <limits>
#include <ostream>
#include <string>
#include <vector>

#include "cudepthfusion/config.hpp"
#include "cudepthfusion/error.hpp"

namespace cudepthfusion {
namespace {

TEST(Config, DefaultsAreValid) { EXPECT_NO_THROW(validate_config(Config{})); }

struct InvalidCase {
  std::string name;
  std::function<void(Config&)> mutate;
  std::string expected_field;
};

// Keeps discovered test names readable instead of a raw byte dump.
void PrintTo(const InvalidCase& invalid_case, std::ostream* os) { *os << invalid_case.name; }

class ConfigInvalid : public ::testing::TestWithParam<InvalidCase> {};

TEST_P(ConfigInvalid, ThrowsNamingTheField) {
  Config config;
  GetParam().mutate(config);
  try {
    validate_config(config);
    FAIL() << "expected ConfigError";
  } catch (const ConfigError& error) {
    EXPECT_NE(std::string(error.what()).find(GetParam().expected_field), std::string::npos)
        << error.what();
  }
}

const double kNan = std::numeric_limits<double>::quiet_NaN();

INSTANTIATE_TEST_SUITE_P(
    AllFields, ConfigInvalid,
    ::testing::Values(
        InvalidCase{"min_zero", [](Config& c) { c.depth.min_m = 0.0; }, "depth.min_m"},
        InvalidCase{"min_nan", [](Config& c) { c.depth.min_m = kNan; }, "depth.min_m"},
        InvalidCase{"max_below_min", [](Config& c) { c.depth.max_m = 0.1; }, "depth.max_m"},
        InvalidCase{"radius_negative", [](Config& c) { c.spatial.radius = -1; }, "spatial.radius"},
        InvalidCase{"radius_too_large", [](Config& c) { c.spatial.radius = kMaxSpatialRadius + 1; },
                    "spatial.radius"},
        InvalidCase{"sigma_xy_zero", [](Config& c) { c.spatial.sigma_xy_px = 0.0; },
                    "spatial.sigma_xy_px"},
        InvalidCase{"sigma_depth_negative", [](Config& c) { c.spatial.sigma_depth_m = -1.0; },
                    "spatial.sigma_depth_m"},
        InvalidCase{"noise_a_negative", [](Config& c) { c.noise.a_m = -0.1; }, "noise.a_m"},
        InvalidCase{"noise_both_zero",
                    [](Config& c) {
                      c.noise.a_m = 0.0;
                      c.noise.b_per_m = 0.0;
                    },
                    "noise.a_m"},
        InvalidCase{"tau_negative", [](Config& c) { c.fusion.tau_abs_m = -0.1; },
                    "fusion.tau_abs_m"},
        InvalidCase{"decay_zero", [](Config& c) { c.fusion.history_decay = 0.0; },
                    "fusion.history_decay"},
        InvalidCase{"decay_above_one", [](Config& c) { c.fusion.history_decay = 1.5; },
                    "fusion.history_decay"},
        InvalidCase{"ratio_zero", [](Config& c) { c.fusion.max_history_ratio = 0.0; },
                    "fusion.max_history_ratio"},
        InvalidCase{"floor_zero", [](Config& c) { c.fusion.variance_floor_m2 = 0.0; },
                    "fusion.variance_floor_m2"},
        InvalidCase{"reference_inf",
                    [](Config& c) {
                      c.fusion.variance_reference_m2 = std::numeric_limits<double>::infinity();
                    },
                    "fusion.variance_reference_m2"},
        InvalidCase{"q0_negative", [](Config& c) { c.fusion.q0_m2 = -1.0; }, "fusion.q0_m2"},
        InvalidCase{"age_too_large",
                    [](Config& c) { c.fusion.max_history_age_frames = kMaxHistoryAgeFrames + 1; },
                    "fusion.max_history_age_frames"},
        InvalidCase{"gap_zero", [](Config& c) { c.reset.max_frame_gap_s = 0.0; },
                    "reset.max_frame_gap_s"}),
    [](const ::testing::TestParamInfo<InvalidCase>& param_info) { return param_info.param.name; });

TEST(Config, BoundaryValuesAreAccepted) {
  Config config;
  config.fusion.history_decay = 1.0;
  config.spatial.radius = 0;
  config.noise.a_m = 0.0;
  config.fusion.max_history_age_frames = 0;
  EXPECT_NO_THROW(validate_config(config));
}

}  // namespace
}  // namespace cudepthfusion
