#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "cpu/measurement.hpp"

namespace cudepthfusion::cpu {
namespace {

TEST(Sanitize, ClassifiesEveryPixel) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const std::vector<float> input = {0.0f, nan, inf, -inf, -1.0f, 0.1f, 0.2f, 5.0f, 8.0f, 8.5f};
  const ImageView<float> view{input.data(), 5, 2};

  std::vector<float> depth;
  std::vector<std::uint8_t> valid;
  const InputStats stats = sanitize_depth(view, DepthRangeConfig{0.2, 8.0}, depth, valid);

  EXPECT_EQ(stats.num_pixels, 10u);
  EXPECT_EQ(stats.num_zero, 1u);
  EXPECT_EQ(stats.num_nonfinite, 3u);
  EXPECT_EQ(stats.num_below_min, 2u);
  EXPECT_EQ(stats.num_above_max, 1u);
  EXPECT_EQ(stats.num_valid, 3u);  // range bounds are inclusive

  const std::vector<std::uint8_t> expected_valid = {0, 0, 0, 0, 0, 0, 1, 1, 1, 0};
  const std::vector<float> expected_depth = {0, 0, 0, 0, 0, 0, 0.2f, 5.0f, 8.0f, 0};
  EXPECT_EQ(valid, expected_valid);
  EXPECT_EQ(depth, expected_depth);
}

TEST(MeasurementVariance, FollowsQuadraticSigmaModel) {
  const std::vector<float> depth = {1.0f, 0.0f, 4.0f};
  const std::vector<std::uint8_t> valid = {1, 0, 1};
  std::vector<float> variance;
  measurement_variance(depth, valid, NoiseConfig{0.002, 0.001}, variance);

  const double sigma1 = 0.002 + 0.001 * 1.0;
  const double sigma4 = 0.002 + 0.001 * 16.0;
  EXPECT_FLOAT_EQ(variance[0], static_cast<float>(sigma1 * sigma1));
  EXPECT_EQ(variance[1], 0.0f);
  EXPECT_FLOAT_EQ(variance[2], static_cast<float>(sigma4 * sigma4));
}

TEST(ConfidenceScore, IsZeroForInvalidAndBoundedForValid) {
  const std::vector<float> variance = {1e-4f, 0.0f, 3e-4f};
  const std::vector<std::uint8_t> valid = {1, 0, 1};
  std::vector<float> confidence;
  confidence_score(variance, valid, 1e-4, confidence);

  EXPECT_FLOAT_EQ(confidence[0], 0.5f);
  EXPECT_EQ(confidence[1], 0.0f);
  EXPECT_FLOAT_EQ(confidence[2], 0.25f);
}

}  // namespace
}  // namespace cudepthfusion::cpu
