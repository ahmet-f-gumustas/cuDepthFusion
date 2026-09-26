#include "cpu/reproject.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>

namespace cudepthfusion::cpu {
namespace {

std::uint32_t float_bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float bits_to_float(std::uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

}  // namespace

RelativePose relative_pose(const RigidTransform& T_world_current,
                           const RigidTransform& T_world_previous) {
  // T_c_p = inverse(T_world_current) * T_world_previous, i.e. R_c^T R_p and R_c^T (t_p - t_c).
  RelativePose pose;
  double rotation[9] = {};
  double translation[3] = {};
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      double sum = 0.0;
      for (int k = 0; k < 3; ++k) {
        sum += T_world_current.at(k, row) * T_world_previous.at(k, col);
      }
      rotation[row * 3 + col] = sum;
    }
  }
  for (int row = 0; row < 3; ++row) {
    double sum = 0.0;
    for (int k = 0; k < 3; ++k) {
      sum += T_world_current.at(k, row) * (T_world_previous.at(k, 3) - T_world_current.at(k, 3));
    }
    translation[row] = sum;
  }
  for (int i = 0; i < 9; ++i) {
    pose.rotation[static_cast<std::size_t>(i)] = static_cast<float>(rotation[i]);
  }
  for (int i = 0; i < 3; ++i) {
    pose.translation[static_cast<std::size_t>(i)] = static_cast<float>(translation[i]);
  }
  pose.translation_norm_m =
      std::sqrt(translation[0] * translation[0] + translation[1] * translation[1] +
                translation[2] * translation[2]);
  const double trace = rotation[0] + rotation[4] + rotation[8];
  pose.rotation_angle_rad = std::acos(std::clamp((trace - 1.0) * 0.5, -1.0, 1.0));
  return pose;
}

std::uint64_t pack_winner(float z_current, std::uint32_t source_index) {
  return (static_cast<std::uint64_t>(float_bits(z_current)) << 32) | source_index;
}

float winner_depth(std::uint64_t key) {
  return bits_to_float(static_cast<std::uint32_t>(key >> 32));
}

std::uint32_t winner_source(std::uint64_t key) { return static_cast<std::uint32_t>(key); }

ProjectionStats project_previous(const std::vector<float>& previous_depth,
                                 const std::vector<std::uint8_t>& previous_valid, int width,
                                 int height, const Intrinsics& previous_intrinsics,
                                 const Intrinsics& current_intrinsics, const RelativePose& pose,
                                 std::vector<std::uint64_t>& winner) {
  winner.assign(previous_depth.size(), kNoWinner);
  ProjectionStats stats;

  const auto inv_fx_p = static_cast<float>(1.0 / previous_intrinsics.fx);
  const auto inv_fy_p = static_cast<float>(1.0 / previous_intrinsics.fy);
  const auto cx_p = static_cast<float>(previous_intrinsics.cx);
  const auto cy_p = static_cast<float>(previous_intrinsics.cy);
  const auto fx_c = static_cast<float>(current_intrinsics.fx);
  const auto fy_c = static_cast<float>(current_intrinsics.fy);
  const auto cx_c = static_cast<float>(current_intrinsics.cx);
  const auto cy_c = static_cast<float>(current_intrinsics.cy);
  const auto& r = pose.rotation;
  const auto& t = pose.translation;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const auto source = static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                          static_cast<std::size_t>(x);
      if (previous_valid[source] == 0) {
        continue;
      }
      const float z = previous_depth[source];
      if (!(z > 0.0f) || !std::isfinite(z)) {
        continue;
      }
      ++stats.candidates;
      const float ray_x = (static_cast<float>(x) - cx_p) * inv_fx_p;
      const float ray_y = (static_cast<float>(y) - cy_p) * inv_fy_p;
      const float px = z * ray_x;
      const float py = z * ray_y;
      const float xc = r[0] * px + r[1] * py + r[2] * z + t[0];
      const float yc = r[3] * px + r[4] * py + r[5] * z + t[1];
      const float zc = r[6] * px + r[7] * py + r[8] * z + t[2];
      if (!std::isfinite(zc) || !(zc > 0.0f)) {
        ++stats.behind_camera;
        continue;
      }
      const float u = fx_c * (xc / zc) + cx_c;
      const float v = fy_c * (yc / zc) + cy_c;
      // Bound the floats before the cast, then bound the rounded index again (spec 4.2).
      if (!std::isfinite(u) || !std::isfinite(v) || u < -1.0f || v < -1.0f ||
          u > static_cast<float>(width) || v > static_cast<float>(height)) {
        ++stats.off_screen;
        continue;
      }
      const int target_x = static_cast<int>(std::floor(u + 0.5f));
      const int target_y = static_cast<int>(std::floor(v + 0.5f));
      if (target_x < 0 || target_x >= width || target_y < 0 || target_y >= height) {
        ++stats.off_screen;
        continue;
      }
      const auto target = static_cast<std::size_t>(target_y) * static_cast<std::size_t>(width) +
                          static_cast<std::size_t>(target_x);
      const std::uint64_t key = pack_winner(zc, static_cast<std::uint32_t>(source));
      winner[target] = std::min(winner[target], key);
    }
  }
  stats.visible = static_cast<std::uint64_t>(std::count_if(
      winner.begin(), winner.end(), [](std::uint64_t key) { return key != kNoWinner; }));
  return stats;
}

float depth_jacobian(const RelativePose& pose, const Intrinsics& previous_intrinsics, int width,
                     std::uint32_t source_index) {
  const int x = static_cast<int>(source_index % static_cast<std::uint32_t>(width));
  const int y = static_cast<int>(source_index / static_cast<std::uint32_t>(width));
  const auto ray_x = static_cast<float>((static_cast<double>(x) - previous_intrinsics.cx) /
                                        previous_intrinsics.fx);
  const auto ray_y = static_cast<float>((static_cast<double>(y) - previous_intrinsics.cy) /
                                        previous_intrinsics.fy);
  return pose.rotation[6] * ray_x + pose.rotation[7] * ray_y + pose.rotation[8];
}

}  // namespace cudepthfusion::cpu
