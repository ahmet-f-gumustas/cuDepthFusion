#include "cudepthfusion/types.hpp"

namespace cudepthfusion {

bool operator==(const Intrinsics& lhs, const Intrinsics& rhs) {
  return lhs.fx == rhs.fx && lhs.fy == rhs.fy && lhs.cx == rhs.cx && lhs.cy == rhs.cy;
}

bool operator!=(const Intrinsics& lhs, const Intrinsics& rhs) { return !(lhs == rhs); }

RigidTransform RigidTransform::identity() {
  RigidTransform transform;
  for (std::size_t i = 0; i < 4; ++i) {
    transform.m[i * 5] = 1.0;
  }
  return transform;
}

const char* to_string(Backend backend) {
  switch (backend) {
    case Backend::kCpu:
      return "cpu";
    case Backend::kCuda:
      return "cuda";
  }
  return "unknown";
}

const char* to_string(ResetReason reason) {
  switch (reason) {
    case ResetReason::kNone:
      return "none";
    case ResetReason::kFirstFrame:
      return "first_frame";
    case ResetReason::kExplicit:
      return "explicit";
    case ResetReason::kResolutionChange:
      return "resolution_change";
    case ResetReason::kIntrinsicsChange:
      return "intrinsics_change";
    case ResetReason::kTimestampNotIncreasing:
      return "timestamp_not_increasing";
    case ResetReason::kFrameGap:
      return "frame_gap";
  }
  return "unknown";
}

const char* to_string(TemporalStatus status) {
  switch (status) {
    case TemporalStatus::kNoHistory:
      return "no_history";
    case TemporalStatus::kFused:
      return "fused";
    case TemporalStatus::kDisabledNoPose:
      return "temporal_disabled_no_pose";
    case TemporalStatus::kDisabledInvalidPose:
      return "temporal_disabled_invalid_pose";
    case TemporalStatus::kNotImplemented:
      return "not_implemented";
  }
  return "unknown";
}

}  // namespace cudepthfusion
