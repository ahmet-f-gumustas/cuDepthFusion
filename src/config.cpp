#include "cudepthfusion/config.hpp"

#include <cmath>
#include <sstream>
#include <string>

#include "cudepthfusion/error.hpp"

namespace cudepthfusion {
namespace {

template <typename T>
[[noreturn]] void fail(const char* field, const char* requirement, T value) {
  std::ostringstream message;
  message << "config field '" << field << "' " << requirement << ", got " << value;
  throw ConfigError(message.str());
}

void require_finite(const char* field, double value) {
  if (!std::isfinite(value)) {
    fail(field, "must be finite", value);
  }
}

void require_positive(const char* field, double value) {
  require_finite(field, value);
  if (!(value > 0.0)) {
    fail(field, "must be > 0", value);
  }
}

void require_non_negative(const char* field, double value) {
  require_finite(field, value);
  if (!(value >= 0.0)) {
    fail(field, "must be >= 0", value);
  }
}

void require_int_range(const char* field, int value, int low, int high) {
  if (value < low || value > high) {
    std::ostringstream requirement;
    requirement << "must be in [" << low << ", " << high << "]";
    fail(field, requirement.str().c_str(), value);
  }
}

void validate_depth(const DepthRangeConfig& depth) {
  require_positive("depth.min_m", depth.min_m);
  require_positive("depth.max_m", depth.max_m);
  if (!(depth.max_m > depth.min_m)) {
    fail("depth.max_m", "must be greater than depth.min_m", depth.max_m);
  }
}

void validate_spatial(const SpatialConfig& spatial) {
  require_int_range("spatial.radius", spatial.radius, 0, kMaxSpatialRadius);
  require_positive("spatial.sigma_xy_px", spatial.sigma_xy_px);
  require_positive("spatial.sigma_depth_m", spatial.sigma_depth_m);
}

void validate_noise(const NoiseConfig& noise) {
  require_non_negative("noise.a_m", noise.a_m);
  require_non_negative("noise.b_per_m", noise.b_per_m);
  if (!(noise.a_m + noise.b_per_m > 0.0)) {
    fail("noise.a_m", "and noise.b_per_m must not both be 0", noise.a_m);
  }
}

void validate_fusion(const FusionConfig& fusion) {
  require_non_negative("fusion.tau_abs_m", fusion.tau_abs_m);
  require_non_negative("fusion.k_sigma", fusion.k_sigma);
  require_positive("fusion.history_decay", fusion.history_decay);
  if (fusion.history_decay > 1.0) {
    fail("fusion.history_decay", "must be in (0, 1]", fusion.history_decay);
  }
  require_positive("fusion.max_history_ratio", fusion.max_history_ratio);
  require_positive("fusion.variance_floor_m2", fusion.variance_floor_m2);
  require_positive("fusion.variance_reference_m2", fusion.variance_reference_m2);
  require_non_negative("fusion.q0_m2", fusion.q0_m2);
  require_non_negative("fusion.q_translation", fusion.q_translation);
  require_non_negative("fusion.q_rotation_m2_per_rad2", fusion.q_rotation_m2_per_rad2);
  require_int_range("fusion.max_history_age_frames", fusion.max_history_age_frames, 0,
                    kMaxHistoryAgeFrames);
}

}  // namespace

void validate_config(const Config& config) {
  validate_depth(config.depth);
  validate_spatial(config.spatial);
  validate_noise(config.noise);
  validate_fusion(config.fusion);
  require_positive("reset.max_frame_gap_s", config.reset.max_frame_gap_s);
}

}  // namespace cudepthfusion
