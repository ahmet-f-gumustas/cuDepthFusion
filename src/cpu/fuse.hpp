#pragma once

#include <cstdint>
#include <vector>

#include "cpu/reproject.hpp"
#include "cudepthfusion/config.hpp"
#include "cudepthfusion/types.hpp"

namespace cudepthfusion::cpu {

// The sanitized (and optionally filtered) current frame.
struct Measurement {
  std::vector<float> depth;
  std::vector<std::uint8_t> valid;
  std::vector<float> variance;  // (a + b z^2)^2, no floor applied yet
};

// What the engine carries between frames. Read-only while a frame is processed.
struct HistoryState {
  std::vector<float> depth;
  std::vector<float> variance;
  std::vector<std::uint8_t> valid;
  std::vector<std::uint16_t> age;
  Intrinsics intrinsics;
  RigidTransform pose;
  int width = 0;
  int height = 0;

  bool empty() const { return depth.empty(); }
  void clear();
};

// The history transported into the current frame, one entry per current pixel.
struct PriorField {
  std::vector<float> depth;     // z_c of the winning source
  std::vector<float> variance;  // j_z^2 * V_previous + Q
  std::vector<std::uint16_t> age;
  std::vector<std::uint8_t> present;

  void clear();
};

// Projects the history, keeps the nearest source per pixel and transports its uncertainty
// (spec 5.3, 5.4). Depth, variance and age of a pixel always come from the same source.
ProjectionStats gather_prior(const HistoryState& history, const Intrinsics& current_intrinsics,
                             const RelativePose& pose, const FusionConfig& config,
                             std::vector<std::uint64_t>& winner, PriorField& prior);

// Compatibility gate and confidence-weighted merge (spec 5.5, 5.6, 5.7). Writes depth,
// validity, variance, source mask and age into ``result``.
FusionStats fuse_frame(const Config& config, const Measurement& measurement,
                       const PriorField& prior, FusionResult& result);

// Largest absolute depth step to a valid 4-neighbour, in metres per pixel. It measures how
// much a half-pixel transport error would cost at each pixel.
void depth_gradient(const Measurement& measurement, int width, int height,
                    std::vector<float>& gradient);

}  // namespace cudepthfusion::cpu
