#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "cpu/fuse.hpp"
#include "cpu/measurement.hpp"

namespace cudepthfusion::cpu {
namespace {

constexpr Intrinsics kIntrinsics{100.0, 100.0, 7.5, 5.5};

Measurement make_measurement(std::vector<float> depth, std::vector<std::uint8_t> valid,
                             const NoiseConfig& noise) {
  Measurement measurement;
  measurement.depth = std::move(depth);
  measurement.valid = std::move(valid);
  measurement_variance(measurement.depth, measurement.valid, noise, measurement.variance);
  return measurement;
}

PriorField make_prior(std::vector<float> depth, std::vector<float> variance,
                      std::vector<std::uint16_t> age) {
  PriorField prior;
  prior.present.assign(depth.size(), 1);
  prior.depth = std::move(depth);
  prior.variance = std::move(variance);
  prior.age = std::move(age);
  return prior;
}

RigidTransform world_pose(double tx, double ty, double tz) {
  RigidTransform pose = RigidTransform::identity();
  pose.m[3] = tx;
  pose.m[7] = ty;
  pose.m[11] = tz;
  return pose;
}

TEST(FuseFrame, CurrentOnlyWhenThereIsNoPrior) {
  const Config config;
  const Measurement measurement = make_measurement({2.0f}, {1}, config.noise);
  FusionResult result;
  result.width = 1;
  result.height = 1;
  const FusionStats stats = fuse_frame(config, measurement, PriorField{}, result);

  EXPECT_EQ(stats.current_only, 1u);
  EXPECT_EQ(result.source_mask[0], static_cast<std::uint8_t>(SourceMask::kCurrent));
  EXPECT_FLOAT_EQ(result.depth_m[0], 2.0f);
  EXPECT_EQ(result.valid_mask[0], 1);
  EXPECT_EQ(result.history_age[0], 0);
}

TEST(FuseFrame, CompatiblePriorIsMergedWithPrecisionWeights) {
  Config config;
  config.fusion.max_history_ratio = 1000.0;  // do not cap in this test
  const Measurement measurement = make_measurement({2.00f}, {1}, config.noise);
  const PriorField prior = make_prior({2.01f}, {1.0e-4f}, {7});
  FusionResult result;
  result.width = 1;
  result.height = 1;
  const FusionStats stats = fuse_frame(config, measurement, prior, result);

  // Read the stored float32 values back, so the expectation uses the same numbers the
  // implementation saw.
  const double variance_current = measurement.variance[0];
  const double precision_current = 1.0 / variance_current;
  const double precision_prior = config.fusion.history_decay / prior.variance[0];
  const double total = precision_current + precision_prior;
  EXPECT_EQ(stats.fused, 1u);
  EXPECT_EQ(result.source_mask[0], static_cast<std::uint8_t>(SourceMask::kFused));
  EXPECT_NEAR(result.depth_m[0],
              (precision_current * 2.00 + precision_prior * prior.depth[0]) / total, 1e-6);
  EXPECT_NEAR(result.variance_m2[0], 1.0 / total, 1e-10);  // the output is float32
  EXPECT_NEAR(stats.mean_prior_weight, precision_prior / total, 1e-9);
  EXPECT_EQ(result.history_age[0], 0);  // a current measurement supports this pixel
}

TEST(FuseFrame, PriorPrecisionIsCappedByMaxHistoryRatio) {
  Config config;
  config.fusion.max_history_ratio = 4.0;
  const Measurement measurement = make_measurement({2.0f}, {1}, config.noise);
  const PriorField prior = make_prior({2.005f}, {1.0e-12f}, {0});  // absurdly confident history
  FusionResult result;
  result.width = 1;
  result.height = 1;
  fuse_frame(config, measurement, prior, result);

  const double precision_current = 1.0 / measurement.variance[0];
  const double precision_prior = config.fusion.max_history_ratio * precision_current;
  const double total = precision_current + precision_prior;
  EXPECT_NEAR(result.depth_m[0], (precision_current * 2.0 + precision_prior * 2.005) / total, 1e-6);
  EXPECT_NEAR(result.variance_m2[0], 1.0 / total, 1e-12);
}

TEST(FuseFrame, FixedPriorWeightReplacesTheAdaptiveWeights) {
  Config config;
  config.fusion.fixed_prior_weight = 0.75;  // baseline B3: the prior always gets 75 %
  const Measurement measurement = make_measurement({2.0f}, {1}, config.noise);
  const PriorField prior = make_prior({2.01f}, {1.0e-9f}, {0});  // variance must not matter
  FusionResult result;
  result.width = 1;
  result.height = 1;
  const FusionStats stats = fuse_frame(config, measurement, prior, result);

  EXPECT_EQ(stats.fused, 1u);
  EXPECT_NEAR(stats.mean_prior_weight, 0.75, 1e-9);
  EXPECT_NEAR(result.depth_m[0], 0.25 * 2.0 + 0.75 * prior.depth[0], 1e-6);
}

TEST(FuseFrame, IncompatiblePriorIsRejectedNeverBlended) {
  const Config config;
  // A new front surface: the current measurement is much nearer than the history.
  const Measurement nearer = make_measurement({1.0f}, {1}, config.noise);
  FusionResult result;
  result.width = 1;
  result.height = 1;
  FusionStats stats = fuse_frame(config, nearer, make_prior({2.0f}, {1.0e-4f}, {0}), result);
  EXPECT_EQ(stats.rejected_current_nearer, 1u);
  EXPECT_EQ(result.source_mask[0], static_cast<std::uint8_t>(SourceMask::kCurrent));
  EXPECT_FLOAT_EQ(result.depth_m[0], 1.0f);

  // A surface that disappeared: the current measurement is much farther.
  const Measurement farther = make_measurement({3.0f}, {1}, config.noise);
  stats = fuse_frame(config, farther, make_prior({1.5f}, {1.0e-4f}, {0}), result);
  EXPECT_EQ(stats.rejected_current_farther, 1u);
  EXPECT_FLOAT_EQ(result.depth_m[0], 3.0f);
}

TEST(FuseFrame, GateWidthFollowsTauAbsAndKSigma) {
  Config config;
  config.fusion.tau_abs_m = 0.0;
  config.fusion.k_sigma = 0.0;  // only exactly equal depths stay compatible
  const Measurement measurement = make_measurement({2.0f}, {1}, config.noise);
  FusionResult result;
  result.width = 1;
  result.height = 1;
  EXPECT_EQ(fuse_frame(config, measurement, make_prior({2.001f}, {1e-4f}, {0}), result).fused, 0u);

  config.fusion.tau_abs_m = 0.01;
  EXPECT_EQ(fuse_frame(config, measurement, make_prior({2.001f}, {1e-4f}, {0}), result).fused, 1u);
}

TEST(FuseFrame, MissingCurrentMeasurementStaysInvalidByDefault) {
  const Config config;
  ASSERT_FALSE(config.fusion.fill_holes);
  const Measurement measurement = make_measurement({0.0f}, {0}, config.noise);
  FusionResult result;
  result.width = 1;
  result.height = 1;
  const FusionStats stats =
      fuse_frame(config, measurement, make_prior({2.0f}, {1e-4f}, {0}), result);

  EXPECT_EQ(stats.invalid, 1u);
  EXPECT_EQ(stats.history_only, 0u);
  EXPECT_EQ(result.valid_mask[0], 0);
  EXPECT_EQ(result.depth_m[0], 0.0f);
}

TEST(FuseFrame, HistoryOnlyModeFillsAgesAndExpires) {
  Config config;
  config.fusion.fill_holes = true;
  config.fusion.max_history_age_frames = 2;
  const Measurement measurement = make_measurement({0.0f}, {0}, config.noise);
  FusionResult result;
  result.width = 1;
  result.height = 1;

  FusionStats stats = fuse_frame(config, measurement, make_prior({2.0f}, {1e-4f}, {0}), result);
  EXPECT_EQ(stats.history_only, 1u);
  EXPECT_EQ(result.source_mask[0], static_cast<std::uint8_t>(SourceMask::kHistoryOnly));
  EXPECT_EQ(result.history_age[0], 1);
  EXPECT_FLOAT_EQ(result.depth_m[0], 2.0f);

  stats = fuse_frame(config, measurement, make_prior({2.0f}, {1e-4f}, {1}), result);
  EXPECT_EQ(result.history_age[0], 2);

  stats = fuse_frame(config, measurement, make_prior({2.0f}, {1e-4f}, {2}), result);
  EXPECT_EQ(stats.history_expired, 1u);
  EXPECT_EQ(result.valid_mask[0], 0);
}

TEST(FuseFrame, VarianceFloorIsApplied) {
  Config config;
  config.noise = NoiseConfig{1e-9, 0.0};
  config.fusion.variance_floor_m2 = 1e-6;
  const Measurement measurement = make_measurement({2.0f}, {1}, config.noise);
  FusionResult result;
  result.width = 1;
  result.height = 1;
  fuse_frame(config, measurement, PriorField{}, result);
  EXPECT_FLOAT_EQ(result.variance_m2[0], 1e-6f);
}

TEST(DepthGradient, ReportsTheLargestStepToAValidNeighbour) {
  Measurement measurement;
  measurement.depth = {1.0f, 1.0f, 2.0f, 0.0f};
  measurement.valid = {1, 1, 1, 0};
  std::vector<float> gradient;
  depth_gradient(measurement, 4, 1, gradient);
  EXPECT_FLOAT_EQ(gradient[0], 0.0f);
  EXPECT_FLOAT_EQ(gradient[1], 1.0f);
  EXPECT_FLOAT_EQ(gradient[2], 1.0f);  // the invalid neighbour is ignored
  EXPECT_FLOAT_EQ(gradient[3], 0.0f);
}

TEST(FuseFrame, SteepSlopesWeakenThePriorInsteadOfTrustingIt) {
  Config config;
  config.fusion.max_history_ratio = 1000.0;
  // A ramp of 0.1 m per pixel: a half-pixel transport error would be 0.05 m.
  Measurement measurement;
  measurement.depth = {2.0f, 2.1f, 2.2f};
  measurement.valid = {1, 1, 1};
  measurement_variance(measurement.depth, measurement.valid, config.noise, measurement.variance);
  // 0.02 m above the measurement: inside the compatibility gate, so the merge runs.
  const PriorField prior = make_prior({2.02f, 2.12f, 2.22f}, {1e-5f, 1e-5f, 1e-5f}, {0, 0, 0});

  FusionResult result;
  result.width = 3;
  result.height = 1;
  const FusionStats with_gradient = fuse_frame(config, measurement, prior, result);

  config.fusion.q_gradient = 0.0;
  FusionResult ignored_slope;
  ignored_slope.width = 3;
  ignored_slope.height = 1;
  const FusionStats without_gradient = fuse_frame(config, measurement, prior, ignored_slope);

  EXPECT_EQ(with_gradient.fused, 3u);
  EXPECT_EQ(without_gradient.fused, 3u);
  EXPECT_LT(with_gradient.mean_prior_weight, 0.5 * without_gradient.mean_prior_weight);
  // The fused depth stays closer to the current measurement on the slope.
  EXPECT_LT(std::fabs(result.depth_m[1] - 2.1f), std::fabs(ignored_slope.depth_m[1] - 2.1f));
}

TEST(GatherPrior, TransportsDepthVarianceAndAgeFromTheSameSource) {
  const Config config;
  HistoryState history;
  const int width = 4;
  const int height = 3;
  const auto count = static_cast<std::size_t>(width * height);
  history.depth.assign(count, 2.0f);
  history.valid.assign(count, 1);
  history.variance.assign(count, 0.0f);
  history.age.assign(count, 0);
  for (std::size_t i = 0; i < count; ++i) {
    history.variance[i] = 1.0e-4f * static_cast<float>(i + 1);
    history.age[i] = static_cast<std::uint16_t>(i);
  }
  history.intrinsics = kIntrinsics;
  history.pose = world_pose(0, 0, 0);
  history.width = width;
  history.height = height;

  std::vector<std::uint64_t> winner;
  PriorField prior;
  const RelativePose pose = relative_pose(world_pose(0, 0, 0), history.pose);
  const ProjectionStats stats =
      gather_prior(history, kIntrinsics, pose, config.fusion, winner, prior);

  EXPECT_EQ(stats.visible, count);
  for (std::size_t i = 0; i < count; ++i) {
    ASSERT_EQ(prior.present[i], 1);
    EXPECT_FLOAT_EQ(prior.depth[i], 2.0f);
    EXPECT_EQ(prior.age[i], history.age[i]);
    // Identity pose: the jacobian is 1, so only the process noise q0 is added.
    EXPECT_NEAR(prior.variance[i], history.variance[i] + config.fusion.q0_m2, 1e-9);
  }
}

TEST(GatherPrior, ProcessNoiseGrowsWithTranslation) {
  const Config config;
  HistoryState history;
  history.depth.assign(1, 2.0f);
  history.valid.assign(1, 1);
  history.variance.assign(1, 1.0e-4f);
  history.age.assign(1, 0);
  history.intrinsics = Intrinsics{100.0, 100.0, 0.0, 0.0};
  history.pose = world_pose(0, 0, 0);
  history.width = 1;
  history.height = 1;

  std::vector<std::uint64_t> winner;
  PriorField prior;
  const RelativePose pose = relative_pose(world_pose(0, 0.0, 0), history.pose);
  gather_prior(history, history.intrinsics, pose, config.fusion, winner, prior);
  const double without_motion = prior.variance[0];

  // Move along z: the same translation length, but the point stays inside the 1x1 image.
  const RelativePose moved = relative_pose(world_pose(0, 0, 0.1), history.pose);
  gather_prior(history, history.intrinsics, moved, config.fusion, winner, prior);
  EXPECT_NEAR(prior.variance[0] - without_motion, config.fusion.q_translation * 0.01, 1e-9);
}

}  // namespace
}  // namespace cudepthfusion::cpu
