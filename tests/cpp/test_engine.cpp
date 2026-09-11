#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "cudepthfusion/build_info.hpp"
#include "cudepthfusion/engine.hpp"
#include "cudepthfusion/error.hpp"

namespace cudepthfusion {
namespace {

constexpr Intrinsics kIntrinsics{481.2, 480.0, 319.5, 239.5};

// Owns the pixel buffer a FrameInput points into.
struct TestFrame {
  std::vector<float> pixels;
  FrameInput input;
};

TestFrame make_frame(int width, int height, float depth_m, double timestamp_s,
                     std::optional<RigidTransform> pose = RigidTransform::identity()) {
  TestFrame frame;
  frame.pixels.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), depth_m);
  frame.input.depth_m = ImageView<float>{frame.pixels.data(), width, height};
  frame.input.intrinsics = kIntrinsics;
  frame.input.T_world_camera = pose;
  frame.input.timestamp_s = timestamp_s;
  return frame;
}

bool has_note_containing(const Diagnostics& diagnostics, const std::string& text) {
  return std::any_of(diagnostics.notes.begin(), diagnostics.notes.end(),
                     [&](const std::string& note) { return note.find(text) != std::string::npos; });
}

TEST(Engine, RejectsInvalidConfig) {
  Config config;
  config.fusion.history_decay = 2.0;
  EXPECT_THROW(DepthFusion(config, Backend::kCpu), ConfigError);
}

TEST(Engine, CudaBackendNeverFallsBackToCpu) {
  // Without CUDA support, or before the P4 kernels exist, requesting CUDA must throw.
  EXPECT_THROW(DepthFusion(Config{}, Backend::kCuda), BackendUnavailableError);
}

TEST(Engine, FirstFrameReturnsSanitizedCurrentMeasurement) {
  const Config config;
  DepthFusion engine(config, Backend::kCpu);
  TestFrame frame = make_frame(13, 7, 2.0f, 0.0);  // sizes not multiples of any block size
  frame.pixels[5] = std::numeric_limits<float>::quiet_NaN();

  const FusionResult result = engine.process(frame.input);

  ASSERT_EQ(result.width, 13);
  ASSERT_EQ(result.height, 7);
  const double sigma = config.noise.a_m + config.noise.b_per_m * 4.0;
  const auto expected_variance = static_cast<float>(sigma * sigma);
  const auto expected_confidence =
      static_cast<float>(1.0 / (1.0 + expected_variance / config.fusion.variance_reference_m2));
  for (std::size_t i = 0; i < frame.pixels.size(); ++i) {
    const bool valid = i != 5;
    EXPECT_EQ(result.valid_mask[i], valid ? 1 : 0);
    EXPECT_EQ(result.depth_m[i], valid ? 2.0f : 0.0f);
    EXPECT_EQ(result.source_mask[i], valid ? 1 : 0);
    EXPECT_EQ(result.history_age[i], 0);
    EXPECT_FLOAT_EQ(result.variance_m2[i], valid ? expected_variance : 0.0f);
    EXPECT_FLOAT_EQ(result.confidence_score[i], valid ? expected_confidence : 0.0f);
  }
  EXPECT_EQ(result.diagnostics.frame_index, 0u);
  EXPECT_EQ(result.diagnostics.reset_reason, ResetReason::kFirstFrame);
  EXPECT_EQ(result.diagnostics.temporal_status, TemporalStatus::kNoHistory);
  EXPECT_EQ(result.diagnostics.input.num_nonfinite, 1u);
  EXPECT_EQ(result.diagnostics.input.num_valid, frame.pixels.size() - 1);
  EXPECT_FALSE(result.diagnostics.spatial_applied);
  EXPECT_TRUE(has_note_containing(result.diagnostics, "spatial filter is not implemented"));
}

TEST(Engine, VarianceFloorApplies) {
  Config config;
  config.noise = NoiseConfig{1e-6, 0.0};
  config.fusion.variance_floor_m2 = 1e-6;
  DepthFusion engine(config, Backend::kCpu);
  const FusionResult result = engine.process(make_frame(2, 2, 1.0f, 0.0).input);
  EXPECT_FLOAT_EQ(result.variance_m2[0], 1e-6f);
}

TEST(Engine, DetectsSequenceDiscontinuities) {
  DepthFusion engine(Config{}, Backend::kCpu);
  auto reason_for = [&](const TestFrame& frame) {
    return engine.process(frame.input).diagnostics.reset_reason;
  };

  EXPECT_EQ(reason_for(make_frame(4, 3, 1.0f, 0.000)), ResetReason::kFirstFrame);
  EXPECT_EQ(reason_for(make_frame(4, 3, 1.0f, 0.033)), ResetReason::kNone);
  EXPECT_EQ(reason_for(make_frame(4, 3, 1.0f, 0.033)), ResetReason::kTimestampNotIncreasing);
  EXPECT_EQ(reason_for(make_frame(4, 3, 1.0f, 0.500)), ResetReason::kFrameGap);
  EXPECT_EQ(reason_for(make_frame(5, 3, 1.0f, 0.533)), ResetReason::kResolutionChange);

  TestFrame recalibrated = make_frame(5, 3, 1.0f, 0.566);
  recalibrated.input.intrinsics.fx = 500.0;
  EXPECT_EQ(reason_for(recalibrated), ResetReason::kIntrinsicsChange);

  engine.reset();
  EXPECT_EQ(reason_for(make_frame(5, 3, 1.0f, 0.0)), ResetReason::kExplicit);
}

TEST(Engine, TemporalStatusReflectsPoseAvailability) {
  DepthFusion engine(Config{}, Backend::kCpu);
  auto status_for = [&](const TestFrame& frame) {
    return engine.process(frame.input).diagnostics.temporal_status;
  };

  EXPECT_EQ(status_for(make_frame(4, 3, 1.0f, 0.00)), TemporalStatus::kNoHistory);
  EXPECT_EQ(status_for(make_frame(4, 3, 1.0f, 0.03)), TemporalStatus::kNotImplemented);
  EXPECT_EQ(status_for(make_frame(4, 3, 1.0f, 0.06, std::nullopt)),
            TemporalStatus::kDisabledNoPose);
  // History from a pose-less frame cannot be reprojected.
  EXPECT_EQ(status_for(make_frame(4, 3, 1.0f, 0.09)), TemporalStatus::kNoHistory);
}

TEST(Engine, RejectsInvalidPoseWithoutThrowing) {
  DepthFusion engine(Config{}, Backend::kCpu);
  engine.process(make_frame(4, 3, 1.0f, 0.00).input);

  RigidTransform scaled = RigidTransform::identity();
  scaled.m[0] = 2.0;
  const FusionResult result = engine.process(make_frame(4, 3, 1.0f, 0.03, scaled).input);

  EXPECT_EQ(result.diagnostics.temporal_status, TemporalStatus::kDisabledInvalidPose);
  EXPECT_TRUE(has_note_containing(result.diagnostics, "pose rejected"));
  EXPECT_EQ(result.diagnostics.input.num_valid, 12u);
}

TEST(Engine, StaticCameraModeTreatsMissingPoseAsIdentity) {
  Config config;
  config.fusion.assume_static_camera = true;
  DepthFusion engine(config, Backend::kCpu);
  engine.process(make_frame(4, 3, 1.0f, 0.00, std::nullopt).input);
  const FusionResult result = engine.process(make_frame(4, 3, 1.0f, 0.03, std::nullopt).input);
  EXPECT_EQ(result.diagnostics.temporal_status, TemporalStatus::kNotImplemented);
}

TEST(Engine, RejectedFrameLeavesStateUnchanged) {
  DepthFusion engine(Config{}, Backend::kCpu);
  engine.process(make_frame(4, 3, 1.0f, 0.00).input);

  TestFrame bad = make_frame(4, 3, 1.0f, 0.03);
  bad.input.intrinsics.fx = 0.0;
  EXPECT_THROW(engine.process(bad.input), InvalidInputError);

  const FusionResult next = engine.process(make_frame(4, 3, 1.0f, 0.03).input);
  EXPECT_EQ(next.diagnostics.frame_index, 1u);
  EXPECT_EQ(next.diagnostics.reset_reason, ResetReason::kNone);
}

TEST(Engine, RejectsMalformedFrames) {
  DepthFusion engine(Config{}, Backend::kCpu);

  TestFrame no_data = make_frame(4, 3, 1.0f, 0.0);
  no_data.input.depth_m.data = nullptr;
  EXPECT_THROW(engine.process(no_data.input), InvalidInputError);

  TestFrame zero_width = make_frame(4, 3, 1.0f, 0.0);
  zero_width.input.depth_m.width = 0;
  EXPECT_THROW(engine.process(zero_width.input), InvalidInputError);

  TestFrame too_wide = make_frame(4, 3, 1.0f, 0.0);
  too_wide.input.depth_m.width = kMaxImageDimension + 1;
  EXPECT_THROW(engine.process(too_wide.input), InvalidInputError);

  TestFrame bad_time = make_frame(4, 3, 1.0f, std::numeric_limits<double>::quiet_NaN());
  EXPECT_THROW(engine.process(bad_time.input), InvalidInputError);
}

TEST(Engine, HandlesDegenerateShapesAndContent) {
  DepthFusion engine(Config{}, Backend::kCpu);
  EXPECT_EQ(engine.process(make_frame(1, 1, 1.0f, 0.0).input).diagnostics.input.num_valid, 1u);
  EXPECT_EQ(engine.process(make_frame(17, 1, 1.0f, 0.1).input).diagnostics.input.num_valid, 17u);
  EXPECT_EQ(engine.process(make_frame(1, 9, 1.0f, 0.2).input).diagnostics.input.num_valid, 9u);

  const FusionResult all_invalid = engine.process(make_frame(1, 9, 0.0f, 0.3).input);
  EXPECT_EQ(all_invalid.diagnostics.input.num_valid, 0u);
  EXPECT_EQ(all_invalid.diagnostics.input.num_zero, 9u);
  EXPECT_TRUE(std::all_of(all_invalid.confidence_score.begin(), all_invalid.confidence_score.end(),
                          [](float c) { return c == 0.0f; }));
}

TEST(Engine, MoveKeepsStateAndConfig) {
  Config config;
  config.fusion.k_sigma = 2.5;
  DepthFusion original(config, Backend::kCpu);
  original.process(make_frame(4, 3, 1.0f, 0.00).input);

  DepthFusion moved(std::move(original));
  EXPECT_EQ(moved.config().fusion.k_sigma, 2.5);
  const FusionResult next = moved.process(make_frame(4, 3, 1.0f, 0.03).input);
  EXPECT_EQ(next.diagnostics.frame_index, 1u);
  EXPECT_EQ(next.diagnostics.reset_reason, ResetReason::kNone);

  DepthFusion assigned(Config{}, Backend::kCpu);
  assigned = std::move(moved);
  EXPECT_EQ(assigned.process(make_frame(4, 3, 1.0f, 0.06).input).diagnostics.frame_index, 2u);
}

TEST(Engine, NoSpatialNoteWhenSpatialDisabled) {
  Config config;
  config.spatial.enabled = false;
  DepthFusion engine(config, Backend::kCpu);
  const FusionResult result = engine.process(make_frame(4, 3, 1.0f, 0.0).input);
  EXPECT_FALSE(has_note_containing(result.diagnostics, "spatial"));
}

TEST(BuildInfo, ReportsVersionAndCompiler) {
  const BuildInfo info = build_info();
  EXPECT_FALSE(info.version.empty());
  EXPECT_FALSE(info.cxx_compiler.empty());
  if (!info.cuda_compiled) {
    EXPECT_TRUE(info.cuda_devices.empty());
  }
}

}  // namespace
}  // namespace cudepthfusion
