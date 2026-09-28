#include "cudepthfusion/engine.hpp"

#include <chrono>
#include <cmath>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "cudepthfusion/error.hpp"
#include "cudepthfusion/geometry.hpp"
#include "pipeline.hpp"

namespace cudepthfusion {
namespace {

// What the engine remembers about the previous frame to detect sequence discontinuities.
struct FrameMeta {
  int width = 0;
  int height = 0;
  Intrinsics intrinsics;
  double timestamp_s = 0.0;
};

std::unique_ptr<detail::Pipeline> make_pipeline(Backend backend) {
  if (backend == Backend::kCpu) {
    return detail::make_cpu_pipeline();
  }
#ifdef CUDEPTHFUSION_WITH_CUDA
  return detail::make_cuda_pipeline();
#else
  throw BackendUnavailableError(
      "backend 'cuda' requested, but this build has no CUDA support; rebuild with "
      "-DCUDEPTHFUSION_ENABLE_CUDA=ON or use backend 'cpu'");
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
  std::unique_ptr<detail::Pipeline> pipeline;
};

DepthFusion::DepthFusion(const Config& config, Backend backend) : impl_(std::make_unique<Impl>()) {
  validate_config(config);
  impl_->pipeline = make_pipeline(backend);  // throws when the backend cannot run
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
    impl_->pipeline->reset();
  }
  diagnostics.temporal_status = decide_temporal_status(
      frame, config.fusion, impl_->pipeline->has_history(), diagnostics.notes);

  detail::FrameRequest request;
  request.frame = &frame;
  request.pose = frame.T_world_camera ? *frame.T_world_camera : RigidTransform::identity();
  request.use_history = diagnostics.temporal_status == TemporalStatus::kFused;
  request.keep_history = pose_is_usable(diagnostics.temporal_status);

  const detail::FrameOutcome outcome = impl_->pipeline->process(config, request, result);
  diagnostics.input = outcome.input;
  diagnostics.fusion = outcome.fusion;
  diagnostics.spatial_applied = outcome.spatial_applied;
  diagnostics.device = outcome.device;

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
  impl_->pipeline->reset();
  impl_->explicit_reset_pending = true;
}

const Config& DepthFusion::config() const { return impl_->config; }

Backend DepthFusion::backend() const { return impl_->backend; }

}  // namespace cudepthfusion
