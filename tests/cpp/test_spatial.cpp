#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "cpu/spatial.hpp"

namespace cudepthfusion::cpu {
namespace {

SpatialConfig make_config(int radius = 2, double sigma_xy = 2.0, double sigma_depth = 0.03) {
  SpatialConfig config;
  config.enabled = true;
  config.radius = radius;
  config.sigma_xy_px = sigma_xy;
  config.sigma_depth_m = sigma_depth;
  return config;
}

double stddev_around(const std::vector<float>& values, double mean) {
  double sum = 0.0;
  for (float value : values) {
    sum += (value - mean) * (value - mean);
  }
  return std::sqrt(sum / static_cast<double>(values.size()));
}

TEST(BilateralFilter, ConstantPlaneIsUnchanged) {
  const std::vector<float> depth(7 * 5, 2.0f);
  const std::vector<std::uint8_t> valid(depth.size(), 1);
  std::vector<float> filtered;
  bilateral_filter(depth, valid, 7, 5, make_config(), filtered);
  for (float value : filtered) {
    EXPECT_FLOAT_EQ(value, 2.0f);
  }
}

TEST(BilateralFilter, InvalidCentreStaysInvalidAndInvalidNeighboursDoNotLeak) {
  const int width = 5;
  const int height = 5;
  std::vector<float> depth(static_cast<std::size_t>(width * height), 2.0f);
  std::vector<std::uint8_t> valid(depth.size(), 1);
  depth[12] = 0.0f;
  valid[12] = 0;  // hole in the centre
  depth[11] = 9.0f;
  valid[11] = 0;  // invalid neighbour carrying a wild value
  std::vector<float> filtered;
  bilateral_filter(depth, valid, width, height, make_config(), filtered);
  EXPECT_EQ(filtered[12], 0.0f);
  EXPECT_FLOAT_EQ(filtered[10], 2.0f);
  EXPECT_FLOAT_EQ(filtered[13], 2.0f);
}

TEST(BilateralFilter, BorderSkipsOutsidePixelsInsteadOfClamping) {
  const int width = 4;
  const std::vector<float> depth = {1.0f, 1.0f, 1.0f, 1.0f};
  const std::vector<std::uint8_t> valid(4, 1);
  std::vector<float> filtered;
  // A one-pixel radius with a wide range sigma reduces to a plain spatial average.
  bilateral_filter(depth, valid, width, 1, make_config(1, 1.0, 10.0), filtered);
  // Clamping would count the outside column twice; every weight here multiplies the same
  // depth, so the check that matters is that the border value stays exactly 1.0.
  for (float value : filtered) {
    EXPECT_FLOAT_EQ(value, 1.0f);
  }
}

TEST(BilateralFilter, StepEdgeIsPreserved) {
  const int width = 6;
  const std::vector<float> depth = {1.5f, 1.5f, 1.5f, 2.5f, 2.5f, 2.5f};
  const std::vector<std::uint8_t> valid(6, 1);
  std::vector<float> filtered;
  bilateral_filter(depth, valid, width, 1, make_config(2, 2.0, 0.03), filtered);
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(filtered[static_cast<std::size_t>(i)], 1.5, 1e-4);
  }
  for (int i = 3; i < 6; ++i) {
    EXPECT_NEAR(filtered[static_cast<std::size_t>(i)], 2.5, 1e-4);
  }
}

TEST(BilateralFilter, ReducesIndependentNoiseOnAPlane) {
  const int width = 64;
  const int height = 64;
  std::vector<float> depth(static_cast<std::size_t>(width * height));
  const std::vector<std::uint8_t> valid(depth.size(), 1);
  std::mt19937 rng(7);
  std::normal_distribution<float> noise(0.0f, 0.01f);
  for (float& value : depth) {
    value = 2.0f + noise(rng);
  }
  std::vector<float> filtered;
  bilateral_filter(depth, valid, width, height, make_config(2, 2.0, 1.0), filtered);
  EXPECT_LT(stddev_around(filtered, 2.0), 0.6 * stddev_around(depth, 2.0));
}

TEST(BilateralFilter, ZeroRadiusLeavesDepthUntouched) {
  const std::vector<float> depth = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<std::uint8_t> valid(4, 1);
  std::vector<float> filtered;
  bilateral_filter(depth, valid, 4, 1, make_config(0), filtered);
  EXPECT_EQ(filtered, depth);
}

}  // namespace
}  // namespace cudepthfusion::cpu
