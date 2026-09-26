#include "cpu/spatial.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace cudepthfusion::cpu {

void bilateral_filter(const std::vector<float>& depth, const std::vector<std::uint8_t>& valid,
                      int width, int height, const SpatialConfig& config,
                      std::vector<float>& filtered) {
  filtered.assign(depth.size(), 0.0f);
  const int radius = config.radius;
  const double spatial_scale = 1.0 / (2.0 * config.sigma_xy_px * config.sigma_xy_px);
  const double range_scale = 1.0 / (2.0 * config.sigma_depth_m * config.sigma_depth_m);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const auto center = static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                          static_cast<std::size_t>(x);
      if (valid[center] == 0) {
        continue;
      }
      const double center_depth = depth[center];
      double weight_sum = 0.0;
      double value_sum = 0.0;
      const int y_first = std::max(y - radius, 0);
      const int y_last = std::min(y + radius, height - 1);
      const int x_first = std::max(x - radius, 0);
      const int x_last = std::min(x + radius, width - 1);
      for (int ny = y_first; ny <= y_last; ++ny) {
        for (int nx = x_first; nx <= x_last; ++nx) {
          const auto neighbour = static_cast<std::size_t>(ny) * static_cast<std::size_t>(width) +
                                 static_cast<std::size_t>(nx);
          if (valid[neighbour] == 0) {
            continue;
          }
          const double dx = nx - x;
          const double dy = ny - y;
          const double dz = depth[neighbour] - center_depth;
          const double weight =
              std::exp(-(dx * dx + dy * dy) * spatial_scale - dz * dz * range_scale);
          weight_sum += weight;
          value_sum += weight * depth[neighbour];
        }
      }
      // The centre always contributes with weight 1, so weight_sum is never zero.
      filtered[center] = static_cast<float>(value_sum / weight_sum);
    }
  }
}

}  // namespace cudepthfusion::cpu
