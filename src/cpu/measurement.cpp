#include "cpu/measurement.hpp"

#include <cmath>

namespace cudepthfusion::cpu {

InputStats sanitize_depth(const ImageView<float>& input, const DepthRangeConfig& range,
                          std::vector<float>& depth_out, std::vector<std::uint8_t>& valid_out) {
  const std::size_t count = input.pixel_count();
  depth_out.assign(count, 0.0f);
  valid_out.assign(count, 0);

  InputStats stats;
  stats.num_pixels = count;
  for (std::size_t i = 0; i < count; ++i) {
    const float z = input.data[i];
    if (!std::isfinite(z)) {
      ++stats.num_nonfinite;
    } else if (z == 0.0f) {
      ++stats.num_zero;
    } else if (static_cast<double>(z) < range.min_m) {
      ++stats.num_below_min;
    } else if (static_cast<double>(z) > range.max_m) {
      ++stats.num_above_max;
    } else {
      depth_out[i] = z;
      valid_out[i] = 1;
      ++stats.num_valid;
    }
  }
  return stats;
}

void measurement_variance(const std::vector<float>& depth, const std::vector<std::uint8_t>& valid,
                          const NoiseConfig& noise, std::vector<float>& variance_out) {
  variance_out.assign(depth.size(), 0.0f);
  for (std::size_t i = 0; i < depth.size(); ++i) {
    if (valid[i] == 0) {
      continue;
    }
    const double z = depth[i];
    const double sigma = noise.a_m + noise.b_per_m * z * z;
    variance_out[i] = static_cast<float>(sigma * sigma);
  }
}

void confidence_score(const std::vector<float>& variance, const std::vector<std::uint8_t>& valid,
                      double variance_reference_m2, std::vector<float>& confidence_out) {
  confidence_out.assign(variance.size(), 0.0f);
  for (std::size_t i = 0; i < variance.size(); ++i) {
    if (valid[i] == 0) {
      continue;
    }
    const double ratio = static_cast<double>(variance[i]) / variance_reference_m2;
    confidence_out[i] = static_cast<float>(1.0 / (1.0 + ratio));
  }
}

}  // namespace cudepthfusion::cpu
