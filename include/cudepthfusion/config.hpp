#pragma once

namespace cudepthfusion {

// Defaults mirror configs/default.yaml. They are starting points, not tuned optima:
// parameters are selected on the validation split and frozen before the test split.

struct DepthRangeConfig {
  double min_m = 0.2;
  double max_m = 8.0;
};

struct SpatialConfig {
  bool enabled = true;
  int radius = 2;
  double sigma_xy_px = 2.0;
  double sigma_depth_m = 0.03;
};

// sigma_measurement(z) = a_m + b_per_m * z^2, variance = sigma^2.
struct NoiseConfig {
  double a_m = 0.002;
  double b_per_m = 0.001;
};

struct FusionConfig {
  double tau_abs_m = 0.015;
  double k_sigma = 3.0;
  double history_decay = 0.95;
  double max_history_ratio = 8.0;
  double variance_floor_m2 = 1e-6;
  double variance_reference_m2 = 1e-4;
  double q0_m2 = 4e-6;
  double q_translation = 1e-3;
  double q_rotation_m2_per_rad2 = 1e-4;
  bool fill_holes = false;
  int max_history_age_frames = 2;
  // Treat a missing pose as identity. Must be chosen explicitly; never inferred.
  bool assume_static_camera = false;
};

struct ResetConfig {
  double max_frame_gap_s = 0.2;
};

struct Config {
  DepthRangeConfig depth;
  SpatialConfig spatial;
  NoiseConfig noise;
  FusionConfig fusion;
  ResetConfig reset;
};

inline constexpr int kMaxSpatialRadius = 16;
// history_age is stored as uint16.
inline constexpr int kMaxHistoryAgeFrames = 65535;

// Throws ConfigError naming the offending field, e.g. "fusion.history_decay".
void validate_config(const Config& config);

}  // namespace cudepthfusion
