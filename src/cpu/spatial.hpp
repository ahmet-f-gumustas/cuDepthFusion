#pragma once

#include <cstdint>
#include <vector>

#include "cudepthfusion/config.hpp"

namespace cudepthfusion::cpu {

// Mask-aware bilateral filter (spec 5.2). Only valid neighbours contribute, out-of-image
// neighbours are skipped rather than clamped (clamping would count a pixel twice), and an
// invalid centre stays invalid. The measurement variance is deliberately left untouched:
// dividing it by the neighbour count would invent confidence.
void bilateral_filter(const std::vector<float>& depth, const std::vector<std::uint8_t>& valid,
                      int width, int height, const SpatialConfig& config,
                      std::vector<float>& filtered);

}  // namespace cudepthfusion::cpu
