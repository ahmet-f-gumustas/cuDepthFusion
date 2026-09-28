#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cudepthfusion {

// Pinhole intrinsics. fx must be positive; fy may be negative (e.g. ICL-NUIM native convention).
struct Intrinsics {
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
};

bool operator==(const Intrinsics& lhs, const Intrinsics& rhs);
bool operator!=(const Intrinsics& lhs, const Intrinsics& rhs);

// Row-major 4x4 rigid transform. T_world_camera maps camera coordinates to world, metres.
struct RigidTransform {
  std::array<double, 16> m{};

  static RigidTransform identity();
  double at(int row, int col) const { return m[static_cast<std::size_t>(row * 4 + col)]; }
};

// Non-owning view of a dense, row-major H x W image.
template <typename T>
struct ImageView {
  const T* data = nullptr;
  int width = 0;
  int height = 0;

  std::size_t pixel_count() const {
    return static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  }
};

struct FrameInput {
  ImageView<float> depth_m;  // camera Z in metres; 0, NaN, Inf and out-of-range become invalid
  Intrinsics intrinsics;
  std::optional<RigidTransform> T_world_camera;  // nullopt: temporal fusion disabled
  double timestamp_s = 0.0;
};

enum class Backend { kCpu, kCuda };

enum class SourceMask : std::uint8_t {
  kInvalid = 0,
  kCurrent = 1,
  kFused = 2,
  kHistoryOnly = 3,
};

// Why history was discarded when this frame arrived.
enum class ResetReason {
  kNone,
  kFirstFrame,
  kExplicit,
  kResolutionChange,
  kIntrinsicsChange,
  kTimestampNotIncreasing,
  kFrameGap,
};

// What the temporal stage did for this frame.
enum class TemporalStatus {
  kNoHistory,
  kFused,
  kDisabledNoPose,
  kDisabledInvalidPose,
};

const char* to_string(Backend backend);
const char* to_string(ResetReason reason);
const char* to_string(TemporalStatus status);

struct InputStats {
  std::uint64_t num_pixels = 0;
  std::uint64_t num_valid = 0;
  std::uint64_t num_zero = 0;  // the "missing" marker of most sensors
  std::uint64_t num_nonfinite = 0;
  std::uint64_t num_below_min = 0;  // non-zero values below depth.min_m, negatives included
  std::uint64_t num_above_max = 0;
};

// Per-frame outcome of the temporal stage (spec 5.3-5.7).
struct FusionStats {
  std::uint64_t prior_candidates = 0;     // valid history pixels that were projected
  std::uint64_t prior_behind_camera = 0;  // z_c <= 0 after transport
  std::uint64_t prior_off_screen = 0;
  std::uint64_t prior_visible = 0;  // pixels that received a transported prior
  std::uint64_t fused = 0;
  std::uint64_t current_only = 0;
  std::uint64_t rejected_current_nearer = 0;   // occlusion or a new front surface
  std::uint64_t rejected_current_farther = 0;  // a surface disappeared or moved away
  std::uint64_t history_only = 0;              // only with fusion.fill_holes
  std::uint64_t history_expired = 0;           // history dropped at max_history_age_frames
  std::uint64_t invalid = 0;
  double mean_prior_weight = 0.0;  // mean P_prior / (P_cur + P_prior) over fused pixels
};

// Measured with CUDA events on the engine's stream. The CPU backend leaves `measured` false:
// a time that was not measured is not reported as zero.
struct DeviceTimings {
  bool measured = false;
  double upload_ms = 0.0;    // host-to-device copy of the input
  double compute_ms = 0.0;   // the kernel chain plus the history update, no host copies
  double download_ms = 0.0;  // device-to-host copy of every output and the counters
  double sanitize_ms = 0.0;
  double bilateral_ms = 0.0;
  double variance_ms = 0.0;
  double reproject_ms = 0.0;  // clearing the z-buffer, projection and gather
  double fuse_ms = 0.0;
  double history_ms = 0.0;  // keeping this frame's result as the next frame's history
};

struct Diagnostics {
  std::uint64_t frame_index = 0;
  ResetReason reset_reason = ResetReason::kNone;
  TemporalStatus temporal_status = TemporalStatus::kNoHistory;
  bool spatial_applied = false;
  InputStats input;
  FusionStats fusion;
  double host_process_ms = 0.0;  // wall clock inside DepthFusion::process()
  DeviceTimings device;
  std::vector<std::string> notes;
};

// Owned outputs. Invalid pixels carry 0 in every field; valid_mask is authoritative.
struct FusionResult {
  int width = 0;
  int height = 0;
  std::vector<float> depth_m;
  std::vector<std::uint8_t> valid_mask;
  std::vector<float> variance_m2;
  std::vector<float> confidence_score;  // display-only score in [0, 1), not a probability
  std::vector<std::uint8_t> source_mask;
  std::vector<std::uint16_t> history_age;
  Diagnostics diagnostics;
};

}  // namespace cudepthfusion
