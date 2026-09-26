#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include "cudepthfusion/types.hpp"

namespace cudepthfusion::cpu {

// Relative transform T_c_p (previous camera -> current camera). It is built in float64 from
// the two world poses and then narrowed to float32: the per-pixel arithmetic below must match
// the CUDA kernel bit for bit, otherwise the winner selection could differ between backends.
struct RelativePose {
  std::array<float, 9> rotation{};  // row-major R_c_p
  std::array<float, 3> translation{};
  double translation_norm_m = 0.0;
  double rotation_angle_rad = 0.0;
};

RelativePose relative_pose(const RigidTransform& T_world_current,
                           const RigidTransform& T_world_previous);

inline constexpr std::uint64_t kNoWinner = std::numeric_limits<std::uint64_t>::max();

// Winner key: [float32 bits of z_c | source pixel index]. For positive finite floats the IEEE
// bit pattern grows with the value, so the smallest key is the nearest surface and, at equal
// depth, the lowest source index. The CUDA backend will get the same result from a 64-bit
// atomicMin. Only positive finite depths may be packed.
std::uint64_t pack_winner(float z_current, std::uint32_t source_index);
float winner_depth(std::uint64_t key);
std::uint32_t winner_source(std::uint64_t key);

struct ProjectionStats {
  std::uint64_t candidates = 0;     // valid previous pixels that were projected
  std::uint64_t behind_camera = 0;  // z_c <= 0 or non-finite
  std::uint64_t off_screen = 0;
  std::uint64_t visible = 0;  // target pixels that received a winner
};

// Forward-projects every valid previous pixel and keeps the nearest one per target pixel.
// Nearest-pixel rounding is floor(coord + 0.5); holes are left empty (spec 5.3).
ProjectionStats project_previous(const std::vector<float>& previous_depth,
                                 const std::vector<std::uint8_t>& previous_valid, int width,
                                 int height, const Intrinsics& previous_intrinsics,
                                 const Intrinsics& current_intrinsics, const RelativePose& pose,
                                 std::vector<std::uint64_t>& winner);

// d(z_c)/d(z_p) for a source pixel under the fixed-ray assumption: third row of R_c_p times the
// source ray (spec 5.4).
float depth_jacobian(const RelativePose& pose, const Intrinsics& previous_intrinsics, int width,
                     std::uint32_t source_index);

}  // namespace cudepthfusion::cpu
