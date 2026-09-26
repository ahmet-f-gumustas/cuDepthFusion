#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "cpu/reproject.hpp"

namespace cudepthfusion::cpu {
namespace {

constexpr int kWidth = 16;
constexpr int kHeight = 12;
constexpr Intrinsics kIntrinsics{100.0, 100.0, 7.5, 5.5};

RigidTransform world_pose(double tx, double ty, double tz, double yaw = 0.0) {
  RigidTransform pose;
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  // clang-format off
  pose.m = {c,   0.0, s,   tx,
            0.0, 1.0, 0.0, ty,
            -s,  0.0, c,   tz,
            0.0, 0.0, 0.0, 1.0};
  // clang-format on
  return pose;
}

struct Frame {
  std::vector<float> depth;
  std::vector<std::uint8_t> valid;
};

Frame constant_depth(float z) {
  const auto count = static_cast<std::size_t>(kWidth * kHeight);
  return Frame{std::vector<float>(count, z), std::vector<std::uint8_t>(count, 1)};
}

Frame single_pixel(int x, int y, float z) {
  Frame frame = constant_depth(0.0f);
  std::fill(frame.valid.begin(), frame.valid.end(), 0);
  const auto index = static_cast<std::size_t>(y * kWidth + x);
  frame.depth[index] = z;
  frame.valid[index] = 1;
  return frame;
}

TEST(WinnerKey, OrdersByDepthThenSourceIndex) {
  EXPECT_FLOAT_EQ(winner_depth(pack_winner(2.5f, 17)), 2.5f);
  EXPECT_EQ(winner_source(pack_winner(2.5f, 17)), 17u);
  EXPECT_LT(pack_winner(0.9f, 4000), pack_winner(1.0f, 0));  // nearer wins
  EXPECT_LT(pack_winner(1.0f, 5), pack_winner(1.0f, 6));     // ties: lower source index
  float previous = 0.0f;
  for (float z : {1e-6f, 0.001f, 0.5f, 1.0f, 2.0f, 100.0f, 1e6f}) {
    EXPECT_LT(pack_winner(previous, 0), pack_winner(z, 0));
    previous = z;
  }
}

TEST(ProjectPrevious, IdentityPoseKeepsEveryPixelAndItsDepth) {
  Frame frame = constant_depth(2.0f);
  for (std::size_t i = 0; i < frame.depth.size(); ++i) {
    frame.depth[i] = 1.0f + 0.01f * static_cast<float>(i);  // a ramp: sources stay distinguishable
  }
  std::vector<std::uint64_t> winner;
  const RelativePose pose = relative_pose(world_pose(0, 0, 0), world_pose(0, 0, 0));
  const ProjectionStats stats = project_previous(frame.depth, frame.valid, kWidth, kHeight,
                                                 kIntrinsics, kIntrinsics, pose, winner);

  EXPECT_EQ(stats.candidates, frame.depth.size());
  EXPECT_EQ(stats.behind_camera, 0u);
  EXPECT_EQ(stats.off_screen, 0u);
  EXPECT_EQ(stats.visible, frame.depth.size());
  for (std::size_t i = 0; i < winner.size(); ++i) {
    ASSERT_NE(winner[i], kNoWinner);
    EXPECT_EQ(winner_source(winner[i]), i);
    EXPECT_EQ(winner_depth(winner[i]), frame.depth[i]);  // depth and source from the same pixel
  }
}

TEST(ProjectPrevious, NegativeFyIsHandledConsistently) {
  const Intrinsics flipped{100.0, -100.0, 7.5, 5.5};
  const Frame frame = constant_depth(2.0f);
  std::vector<std::uint64_t> winner;
  const RelativePose pose = relative_pose(world_pose(0, 0, 0), world_pose(0, 0, 0));
  project_previous(frame.depth, frame.valid, kWidth, kHeight, flipped, flipped, pose, winner);
  for (std::size_t i = 0; i < winner.size(); ++i) {
    ASSERT_NE(winner[i], kNoWinner);
    EXPECT_EQ(winner_source(winner[i]), i);
  }
}

TEST(ProjectPrevious, LateralCameraMotionShiftsByTheAnalyticColumnCount) {
  // The camera moves +0.04 m in x, so a fixed point moves -0.04 m in camera x:
  // the column shift is fx * (-0.04) / 2 = -2.
  const Frame frame = constant_depth(2.0f);
  std::vector<std::uint64_t> winner;
  const RelativePose pose = relative_pose(world_pose(0.04, 0, 0), world_pose(0, 0, 0));
  const ProjectionStats stats = project_previous(frame.depth, frame.valid, kWidth, kHeight,
                                                 kIntrinsics, kIntrinsics, pose, winner);
  EXPECT_EQ(stats.off_screen, static_cast<std::uint64_t>(2 * kHeight));  // two columns leave
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 2; x < kWidth; ++x) {
      const auto target = static_cast<std::size_t>(y * kWidth + (x - 2));
      ASSERT_NE(winner[target], kNoWinner) << "x=" << x << " y=" << y;
      EXPECT_EQ(winner_source(winner[target]), static_cast<std::uint32_t>(y * kWidth + x));
      EXPECT_FLOAT_EQ(winner_depth(winner[target]), 2.0f);
    }
  }
}

TEST(ProjectPrevious, ForwardMotionChangesDepthByTheTranslation) {
  const Frame frame = constant_depth(2.0f);
  std::vector<std::uint64_t> winner;
  const RelativePose pose = relative_pose(world_pose(0, 0, 0.5), world_pose(0, 0, 0));
  project_previous(frame.depth, frame.valid, kWidth, kHeight, kIntrinsics, kIntrinsics, pose,
                   winner);
  for (std::uint64_t key : winner) {
    if (key != kNoWinner) {
      EXPECT_FLOAT_EQ(winner_depth(key), 1.5f);
    }
  }
}

TEST(ProjectPrevious, PureRotationMovesPixelsIndependentlyOfDepth) {
  const double yaw = 0.1;
  Frame frame = constant_depth(0.0f);
  std::fill(frame.valid.begin(), frame.valid.end(), 0);
  const int row_near = 4;
  const int row_far = 7;
  const int column = 10;
  frame.depth[static_cast<std::size_t>(row_near * kWidth + column)] = 1.0f;
  frame.valid[static_cast<std::size_t>(row_near * kWidth + column)] = 1;
  frame.depth[static_cast<std::size_t>(row_far * kWidth + column)] = 3.0f;
  frame.valid[static_cast<std::size_t>(row_far * kWidth + column)] = 1;

  std::vector<std::uint64_t> winner;
  const RelativePose pose = relative_pose(world_pose(0, 0, 0, yaw), world_pose(0, 0, 0));
  project_previous(frame.depth, frame.valid, kWidth, kHeight, kIntrinsics, kIntrinsics, pose,
                   winner);

  // Rotating the camera by yaw subtracts yaw from the ray angle, whatever the depth is.
  const double ray = (column - kIntrinsics.cx) / kIntrinsics.fx;
  const double rotated = (ray - std::tan(yaw)) / (1.0 + ray * std::tan(yaw));
  const int expected_column =
      static_cast<int>(std::floor(kIntrinsics.fx * rotated + kIntrinsics.cx + 0.5));
  int found = 0;
  for (std::size_t i = 0; i < winner.size(); ++i) {
    if (winner[i] != kNoWinner) {
      EXPECT_EQ(static_cast<int>(i % kWidth), expected_column);
      ++found;
    }
  }
  EXPECT_EQ(found, 2);  // both depths land in the same column
}

TEST(ProjectPrevious, PointsBehindTheCameraAndOffScreenAreCountedNotProjected) {
  const Frame frame = constant_depth(2.0f);
  std::vector<std::uint64_t> winner;

  const RelativePose behind = relative_pose(world_pose(0, 0, 3.0), world_pose(0, 0, 0));
  ProjectionStats stats = project_previous(frame.depth, frame.valid, kWidth, kHeight, kIntrinsics,
                                           kIntrinsics, behind, winner);
  EXPECT_EQ(stats.behind_camera, frame.depth.size());
  EXPECT_EQ(stats.visible, 0u);

  const RelativePose far_left = relative_pose(world_pose(1.0, 0, 0), world_pose(0, 0, 0));
  stats = project_previous(frame.depth, frame.valid, kWidth, kHeight, kIntrinsics, kIntrinsics,
                           far_left, winner);
  EXPECT_EQ(stats.off_screen, frame.depth.size());
  EXPECT_EQ(stats.visible, 0u);
}

TEST(ProjectPrevious, NearestSourceWinsWhenTwoPixelsLandOnOneTarget) {
  // With t_c_p = (0.04, 0, 0) the column shift is fx * 0.04 / z = 4 / z:
  // (x=4, z=1) and (x=6, z=2) both land on column 8.
  Frame frame = single_pixel(4, 5, 1.0f);
  frame.depth[static_cast<std::size_t>(5 * kWidth + 6)] = 2.0f;
  frame.valid[static_cast<std::size_t>(5 * kWidth + 6)] = 1;

  std::vector<std::uint64_t> winner;
  const RelativePose pose = relative_pose(world_pose(-0.04, 0, 0), world_pose(0, 0, 0));
  const ProjectionStats stats = project_previous(frame.depth, frame.valid, kWidth, kHeight,
                                                 kIntrinsics, kIntrinsics, pose, winner);

  EXPECT_EQ(stats.candidates, 2u);
  EXPECT_EQ(stats.visible, 1u);
  const auto target = static_cast<std::size_t>(5 * kWidth + 8);
  ASSERT_NE(winner[target], kNoWinner);
  EXPECT_EQ(winner_source(winner[target]), static_cast<std::uint32_t>(5 * kWidth + 4));
  EXPECT_FLOAT_EQ(winner_depth(winner[target]), 1.0f);
}

TEST(ProjectPrevious, HalfPixelCoordinatesRoundUp) {
  // Column shift fx * 0.01 / 2 = 0.5, so pixel 6 lands exactly at 6.5 and rounds to 7.
  const Frame frame = single_pixel(6, 5, 2.0f);
  std::vector<std::uint64_t> winner;
  const RelativePose pose = relative_pose(world_pose(-0.01, 0, 0), world_pose(0, 0, 0));
  project_previous(frame.depth, frame.valid, kWidth, kHeight, kIntrinsics, kIntrinsics, pose,
                   winner);
  EXPECT_NE(winner[static_cast<std::size_t>(5 * kWidth + 7)], kNoWinner);
  EXPECT_EQ(winner[static_cast<std::size_t>(5 * kWidth + 6)], kNoWinner);
}

TEST(ProjectPrevious, NonPositiveAndNonFiniteDepthsNeverEnterTheKey) {
  Frame frame = constant_depth(2.0f);
  frame.depth[0] = 0.0f;  // marked valid by mistake
  frame.depth[1] = std::numeric_limits<float>::quiet_NaN();
  frame.depth[2] = -1.0f;
  std::vector<std::uint64_t> winner;
  const RelativePose pose = relative_pose(world_pose(0, 0, 0), world_pose(0, 0, 0));
  const ProjectionStats stats = project_previous(frame.depth, frame.valid, kWidth, kHeight,
                                                 kIntrinsics, kIntrinsics, pose, winner);
  EXPECT_EQ(stats.candidates, frame.depth.size() - 3);
  EXPECT_EQ(winner[0], kNoWinner);
  EXPECT_EQ(winner[1], kNoWinner);
  EXPECT_EQ(winner[2], kNoWinner);
}

TEST(DepthJacobian, IsOneForIdentityAndMatchesTheThirdRow) {
  const RelativePose identity = relative_pose(world_pose(0, 0, 0), world_pose(0, 0, 0));
  EXPECT_FLOAT_EQ(depth_jacobian(identity, kIntrinsics, kWidth, 0), 1.0f);
  EXPECT_FLOAT_EQ(depth_jacobian(identity, kIntrinsics, kWidth, 37), 1.0f);

  const double yaw = 0.2;
  const RelativePose rotated = relative_pose(world_pose(0, 0, 0, yaw), world_pose(0, 0, 0));
  const std::uint32_t index = 3 * kWidth + 11;
  const double ray_x = (11 - kIntrinsics.cx) / kIntrinsics.fx;
  // R_c_p = R_y(-yaw): third row is (sin(yaw), 0, cos(yaw)).
  const double expected = std::sin(yaw) * ray_x + std::cos(yaw);
  EXPECT_NEAR(depth_jacobian(rotated, kIntrinsics, kWidth, index), expected, 1e-6);
}

TEST(RelativePose, ComposesInverseCurrentWithPrevious) {
  const RelativePose pose = relative_pose(world_pose(0.1, -0.2, 0.3, 0.25), world_pose(0, 0, 0));
  EXPECT_NEAR(pose.rotation_angle_rad, 0.25, 1e-9);
  EXPECT_NEAR(pose.translation_norm_m, std::sqrt(0.01 + 0.04 + 0.09), 1e-9);
}

}  // namespace
}  // namespace cudepthfusion::cpu
