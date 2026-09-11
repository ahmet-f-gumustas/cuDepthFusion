#include "cudepthfusion/engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>

#include "cpu/measurement.hpp"
#include "cudepthfusion/error.hpp"
#include "cudepthfusion/geometry.hpp"
#ifdef CUDEPTHFUSION_WITH_CUDA
#include "cuda/device_info.hpp"
#endif

namespace cudepthfusion {
namespace {

// What the engine remembers about the previous frame to detect sequence discontinuities.
struct FrameMeta {
  int width = 0;
  int height = 0;
  Intrinsics intrinsics;
  double timestamp_s = 0.0;
  bool has_usable_pose = false;
};

void require_cuda_backend() {
#ifndef CUDEPTHFUSION_WITH_CUDA
  throw BackendUnavailableError(
      "backend 'cuda' requested, but this build has no CUDA support; rebuild with "
      "-DCUDEPTHFUSION_ENABLE_CUDA=ON or use backend 'cpu'");
#else
  const cuda::DeviceQuery query = cuda::query_devices();
  if (query.devices.empty()) {
    throw BackendUnavailableError(
        "backend 'cuda' requested, but no usable CUDA device was found: " + query.error);
  }
  throw BackendUnavailableError(
      "backend 'cuda' is not implemented yet (planned for phase P4); use backend 'cpu'");
#endif
}

void validate_frame(const FrameInput& frame) {
  const ImageView<float>& depth = frame.depth_m;
  if (depth.data == nullptr) {
    throw InvalidInputError("depth_m has no data");
  }
  if (depth.width < 1 || depth.height < 1 || depth.width > kMaxImageDimension ||
      depth.height > kMaxImageDimension) {
    std::ostringstream message;
    message << "depth_m shape (" << depth.height << ", " << depth.width
            << ") is outside the supported range [1, " << kMaxImageDimension << "] per dimension";
    throw InvalidInputError(message.str());
  }
  validate_intrinsics(frame.intrinsics);
  if (!std::isfinite(frame.timestamp_s)) {
    std::ostringstream message;
    message << "timestamp_s must be finite, got " << frame.timestamp_s;
    throw InvalidInputError(message.str());
  }
}

ResetReason detect_reset(const std::optional<FrameMeta>& previous, bool explicit_reset_pending,
                         const FrameInput& frame, double max_frame_gap_s) {
  if (!previous) {
    return explicit_reset_pending ? ResetReason::kExplicit : ResetReason::kFirstFrame;
  }
  if (frame.depth_m.width != previous->width || frame.depth_m.height != previous->height) {
    return ResetReason::kResolutionChange;
  }
  if (frame.intrinsics != previous->intrinsics) {
    return ResetReason::kIntrinsicsChange;
  }
  if (!(frame.timestamp_s > previous->timestamp_s)) {
    return ResetReason::kTimestampNotIncreasing;
  }
  if (frame.timestamp_s - previous->timestamp_s > max_frame_gap_s) {
    return ResetReason::kFrameGap;
  }
  return ResetReason::kNone;
}

// A missing pose is identity only when the user explicitly chose the static-camera mode.
TemporalStatus decide_temporal_status(const FrameInput& frame, const FusionConfig& fusion,
                                      ResetReason reset_reason,
                                      const std::optional<FrameMeta>& previous,
                                      std::vector<std::string>& notes) {
  if (!frame.T_world_camera && !fusion.assume_static_camera) {
    return TemporalStatus::kDisabledNoPose;
  }
  if (frame.T_world_camera) {
    const std::string pose_error = rigid_transform_error(*frame.T_world_camera);
    if (!pose_error.empty()) {
      notes.push_back("pose rejected: " + pose_error);
      return TemporalStatus::kDisabledInvalidPose;
    }
  }
  if (reset_reason != ResetReason::kNone || !previous->has_usable_pose) {
    return TemporalStatus::kNoHistory;
  }
  notes.emplace_back(
      "temporal fusion is not implemented yet (phase P3); output is the current measurement");
  return TemporalStatus::kNotImplemented;
}

// Current-only output: variance floored as in the fusion rule, confidence derived from it.
void fill_current_only(const Config& config, FusionResult& result) {
  cpu::measurement_variance(result.depth_m, result.valid_mask, config.noise, result.variance_m2);
  const auto floor = static_cast<float>(config.fusion.variance_floor_m2);
  for (std::size_t i = 0; i < result.variance_m2.size(); ++i) {
    if (result.valid_mask[i] != 0) {
      result.variance_m2[i] = std::max(result.variance_m2[i], floor);
    }
  }
  cpu::confidence_score(result.variance_m2, result.valid_mask, config.fusion.variance_reference_m2,
                        result.confidence_score);

  result.source_mask.assign(result.valid_mask.size(),
                            static_cast<std::uint8_t>(SourceMask::kInvalid));
  for (std::size_t i = 0; i < result.valid_mask.size(); ++i) {
    if (result.valid_mask[i] != 0) {
      result.source_mask[i] = static_cast<std::uint8_t>(SourceMask::kCurrent);
    }
  }
  result.history_age.assign(result.valid_mask.size(), 0);
}

}  // namespace

struct DepthFusion::Impl {
  Config config;
  Backend backend = Backend::kCpu;
  std::mutex mutex;
  std::optional<FrameMeta> previous;
  bool explicit_reset_pending = false;
  std::uint64_t frames_processed = 0;
};

DepthFusion::DepthFusion(const Config& config, Backend backend) : impl_(std::make_unique<Impl>()) {
  validate_config(config);
  if (backend == Backend::kCuda) {
    require_cuda_backend();
  }
  impl_->config = config;
  impl_->backend = backend;
}

DepthFusion::~DepthFusion() = default;
DepthFusion::DepthFusion(DepthFusion&&) noexcept = default;
DepthFusion& DepthFusion::operator=(DepthFusion&&) noexcept = default;

FusionResult DepthFusion::process(const FrameInput& frame) {
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto start = std::chrono::steady_clock::now();
  const Config& config = impl_->config;

  // Validate before touching state so a rejected frame leaves the engine unchanged.
  validate_frame(frame);

  FusionResult result;
  result.width = frame.depth_m.width;
  result.height = frame.depth_m.height;
  Diagnostics& diagnostics = result.diagnostics;
  diagnostics.frame_index = impl_->frames_processed;
  diagnostics.reset_reason = detect_reset(impl_->previous, impl_->explicit_reset_pending, frame,
                                          config.reset.max_frame_gap_s);
  diagnostics.temporal_status = decide_temporal_status(
      frame, config.fusion, diagnostics.reset_reason, impl_->previous, diagnostics.notes);

  diagnostics.input =
      cpu::sanitize_depth(frame.depth_m, config.depth, result.depth_m, result.valid_mask);
  if (config.spatial.enabled) {
    diagnostics.notes.emplace_back(
        "spatial filter is not implemented yet (phase P3); output is unfiltered");
  }
  fill_current_only(config, result);

  const bool pose_usable = diagnostics.temporal_status != TemporalStatus::kDisabledNoPose &&
                           diagnostics.temporal_status != TemporalStatus::kDisabledInvalidPose;
  impl_->previous =
      FrameMeta{result.width, result.height, frame.intrinsics, frame.timestamp_s, pose_usable};
  impl_->explicit_reset_pending = false;
  ++impl_->frames_processed;

  const auto elapsed = std::chrono::steady_clock::now() - start;
  diagnostics.host_process_ms = std::chrono::duration<double, std::milli>(elapsed).count();
  return result;
}

void DepthFusion::reset() {
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->previous.reset();
  impl_->explicit_reset_pending = true;
}

const Config& DepthFusion::config() const { return impl_->config; }

Backend DepthFusion::backend() const { return impl_->backend; }

}  // namespace cudepthfusion
