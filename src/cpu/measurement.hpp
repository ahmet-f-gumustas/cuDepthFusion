#pragma once

#include <cstdint>
#include <vector>

#include "cudepthfusion/config.hpp"
#include "cudepthfusion/types.hpp"

// CPU reference for the per-pixel measurement stages. Output vectors are resized here.
namespace cudepthfusion::cpu {

// Valid iff finite and inside [range.min_m, range.max_m]. Invalid pixels are written as 0.
InputStats sanitize_depth(const ImageView<float>& input, const DepthRangeConfig& range,
                          std::vector<float>& depth_out, std::vector<std::uint8_t>& valid_out);

// variance = (a_m + b_per_m * z^2)^2 for valid pixels, 0 otherwise. No floor applied.
void measurement_variance(const std::vector<float>& depth, const std::vector<std::uint8_t>& valid,
                          const NoiseConfig& noise, std::vector<float>& variance_out);

// confidence = 1 / (1 + V / variance_reference_m2) for valid pixels, 0 otherwise.
void confidence_score(const std::vector<float>& variance, const std::vector<std::uint8_t>& valid,
                      double variance_reference_m2, std::vector<float>& confidence_out);

}  // namespace cudepthfusion::cpu
