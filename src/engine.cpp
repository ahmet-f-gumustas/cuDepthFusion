#include "cudepthfusion/engine.hpp"

#include <chrono>
#include <cmath>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "cpu/fuse.hpp"
#include "cpu/measurement.hpp"
#include "cpu/reproject.hpp"
#include "cpu/spatial.hpp"
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
                                      bool history_available, std::vector<std::string>& notes) {
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
  return history_available ? TemporalStatus::kFused : TemporalStatus::kNoHistory;
}

bool pose_is_usable(TemporalStatus status) {
  return status != TemporalStatus::kDisabledNoPose &&
         status != TemporalStatus::kDisabledInvalidPose;
}

}  // namespace

struct DepthFusion::Impl {
  Config config;
  Backend backend = Backend::kCpu;
  std::mutex mutex;
  std::optional<FrameMeta> previous;
  bool explicit_reset_pending = false;
  std::uint64_t frames_processed = 0;

  // Frame state: the history is read-only while a frame is processed and replaced at the end.
  cpu::HistoryState history;
  cpu::Measurement measurement;
  cpu::PriorField prior;
  std::vector<std::uint64_t> winner;
  std::vector<float> scratch;
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
  if (diagnostics.reset_reason != ResetReason::kNone) {
    impl_->history.clear();
  }
  diagnostics.temporal_status =
      decide_temporal_status(frame, config.fusion, !impl_->history.empty(), diagnostics.notes);

  cpu::Measurement& measurement = impl_->measurement;
  diagnostics.input =
      cpu::sanitize_depth(frame.depth_m, config.depth, measurement.depth, measurement.valid);
  if (config.spatial.enabled && config.spatial.radius > 0) {
    cpu::bilateral_filter(measurement.depth, measurement.valid, result.width, result.height,
                          config.spatial, impl_->scratch);
    measurement.depth.swap(impl_->scratch);
    diagnostics.spatial_applied = true;
  }
  cpu::measurement_variance(measurement.depth, measurement.valid, config.noise,
                            measurement.variance);

  const RigidTransform pose =
      frame.T_world_camera ? *frame.T_world_camera : RigidTransform::identity();
  impl_->prior.clear();
  cpu::ProjectionStats projection;
  if (diagnostics.temporal_status == TemporalStatus::kFused) {
    const cpu::RelativePose relative = cpu::relative_pose(pose, impl_->history.pose);
    projection = cpu::gather_prior(impl_->history, frame.intrinsics, relative, config.fusion,
                                   impl_->winner, impl_->prior);
  }

  diagnostics.fusion = cpu::fuse_frame(config, measurement, impl_->prior, result);
  diagnostics.fusion.prior_candidates = projection.candidates;
  diagnostics.fusion.prior_behind_camera = projection.behind_camera;
  diagnostics.fusion.prior_off_screen = projection.off_screen;
  diagnostics.fusion.prior_visible = projection.visible;
  cpu::confidence_score(result.variance_m2, result.valid_mask, config.fusion.variance_reference_m2,
                        result.confidence_score);

  if (pose_is_usable(diagnostics.temporal_status)) {
    impl_->history.depth = result.depth_m;
    impl_->history.variance = result.variance_m2;
    impl_->history.valid = result.valid_mask;
    impl_->history.age = result.history_age;
    impl_->history.intrinsics = frame.intrinsics;
    impl_->history.pose = pose;
    impl_->history.width = result.width;
    impl_->history.height = result.height;
  } else {
    impl_->history.clear();  // without a pose this frame can never be reprojected
  }
  impl_->previous = FrameMeta{result.width, result.height, frame.intrinsics, frame.timestamp_s};
  impl_->explicit_reset_pending = false;
  ++impl_->frames_processed;

  const auto elapsed = std::chrono::steady_clock::now() - start;
  diagnostics.host_process_ms = std::chrono::duration<double, std::milli>(elapsed).count();
  return result;
}

void DepthFusion::reset() {
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->previous.reset();
  impl_->history.clear();
  impl_->explicit_reset_pending = true;
}

const Config& DepthFusion::config() const { return impl_->config; }

Backend DepthFusion::backend() const { return impl_->backend; }

}  // namespace cudepthfusion
