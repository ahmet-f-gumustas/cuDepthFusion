#include "cpu/fuse.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace cudepthfusion::cpu {

void HistoryState::clear() {
  depth.clear();
  variance.clear();
  valid.clear();
  age.clear();
  width = 0;
  height = 0;
}

void PriorField::clear() {
  depth.clear();
  variance.clear();
  age.clear();
  present.clear();
}

ProjectionStats gather_prior(const HistoryState& history, const Intrinsics& current_intrinsics,
                             const RelativePose& pose, const FusionConfig& config,
                             std::vector<std::uint64_t>& winner, PriorField& prior) {
  const std::size_t count = history.depth.size();
  prior.depth.assign(count, 0.0f);
  prior.variance.assign(count, 0.0f);
  prior.age.assign(count, 0);
  prior.present.assign(count, 0);

  const ProjectionStats stats =
      project_previous(history.depth, history.valid, history.width, history.height,
                       history.intrinsics, current_intrinsics, pose, winner);

  // Process noise: a conservative model of what transporting the history costs us.
  const double process_noise =
      config.q0_m2 + config.q_translation * pose.translation_norm_m * pose.translation_norm_m +
      config.q_rotation_m2_per_rad2 * pose.rotation_angle_rad * pose.rotation_angle_rad;

  for (std::size_t target = 0; target < count; ++target) {
    const std::uint64_t key = winner[target];
    if (key == kNoWinner) {
      continue;
    }
    const std::uint32_t source = winner_source(key);
    const float jacobian = depth_jacobian(pose, history.intrinsics, history.width, source);
    const double transported =
        static_cast<double>(jacobian) * jacobian * history.variance[source] + process_noise;
    prior.depth[target] = winner_depth(key);
    prior.variance[target] = static_cast<float>(transported);
    prior.age[target] = history.age[source];
    prior.present[target] = 1;
  }
  return stats;
}

void depth_gradient(const Measurement& measurement, int width, int height,
                    std::vector<float>& gradient) {
  gradient.assign(measurement.depth.size(), 0.0f);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const auto index = static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                         static_cast<std::size_t>(x);
      if (measurement.valid[index] == 0) {
        continue;
      }
      float largest = 0.0f;
      const int offsets[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
      for (const auto& offset : offsets) {
        const int nx = x + offset[0];
        const int ny = y + offset[1];
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
          continue;
        }
        const auto neighbour = static_cast<std::size_t>(ny) * static_cast<std::size_t>(width) +
                               static_cast<std::size_t>(nx);
        if (measurement.valid[neighbour] == 0) {
          continue;
        }
        largest =
            std::max(largest, std::fabs(measurement.depth[neighbour] - measurement.depth[index]));
      }
      gradient[index] = largest;
    }
  }
}

FusionStats fuse_frame(const Config& config, const Measurement& measurement,
                       const PriorField& prior, FusionResult& result) {
  const std::size_t count = measurement.depth.size();
  if (static_cast<std::size_t>(result.width) * static_cast<std::size_t>(result.height) != count) {
    throw std::logic_error("fuse_frame: result.width/height do not match the measurement size");
  }
  result.depth_m.assign(count, 0.0f);
  result.valid_mask.assign(count, 0);
  result.variance_m2.assign(count, 0.0f);
  result.source_mask.assign(count, static_cast<std::uint8_t>(SourceMask::kInvalid));
  result.history_age.assign(count, 0);

  const FusionConfig& fusion = config.fusion;
  const double floor_variance = fusion.variance_floor_m2;
  const bool has_prior = !prior.present.empty();
  FusionStats stats;
  double prior_weight_sum = 0.0;

  // Nearest-pixel transport is only exact on a locally flat surface; on a slope it can be
  // half a pixel off. Charge that to the prior's variance instead of trusting it (spec 5.4).
  std::vector<float> gradient;
  if (has_prior && fusion.q_gradient > 0.0) {
    depth_gradient(measurement, result.width, result.height, gradient);
  }

  for (std::size_t i = 0; i < count; ++i) {
    const bool current_valid = measurement.valid[i] != 0;
    const bool prior_present = has_prior && prior.present[i] != 0;

    if (current_valid) {
      const double depth_current = measurement.depth[i];
      const double variance_current = std::max<double>(measurement.variance[i], floor_variance);
      if (!prior_present) {
        result.depth_m[i] = static_cast<float>(depth_current);
        result.variance_m2[i] = static_cast<float>(variance_current);
        result.source_mask[i] = static_cast<std::uint8_t>(SourceMask::kCurrent);
        ++stats.current_only;
      } else {
        const double depth_prior = prior.depth[i];
        // The gate asks whether this is the same surface, so it uses only the transported
        // uncertainty: widening it with the sampling term would stop rejecting the history at
        // exactly the depth edges where occlusions happen. The merge below does account for
        // the sampling term, because there it only decides how much the prior is trusted.
        const double variance_prior = std::max<double>(prior.variance[i], floor_variance);
        const double slope = gradient.empty() ? 0.0 : gradient[i];
        const double variance_prior_merge =
            std::max<double>(prior.variance[i] + fusion.q_gradient * slope * slope, floor_variance);
        const double tau =
            fusion.tau_abs_m + fusion.k_sigma * std::sqrt(variance_current + variance_prior);
        if (std::fabs(depth_current - depth_prior) <= tau) {
          const double precision_current = 1.0 / variance_current;
          // The caps keep re-used and spatially filtered observations from piling up into
          // overconfidence; the result is not a full Bayesian posterior (spec 5.6).
          // With fixed_prior_weight the weights ignore the variances entirely: that is
          // baseline B3, which isolates what the adaptive confidence is worth.
          const double precision_prior =
              fusion.fixed_prior_weight > 0.0
                  ? precision_current * fusion.fixed_prior_weight /
                        (1.0 - fusion.fixed_prior_weight)
                  : std::min(fusion.history_decay / variance_prior_merge,
                             fusion.max_history_ratio * precision_current);
          const double total = precision_current + precision_prior;
          result.depth_m[i] = static_cast<float>(
              (precision_current * depth_current + precision_prior * depth_prior) / total);
          result.variance_m2[i] = static_cast<float>(std::max(1.0 / total, floor_variance));
          result.source_mask[i] = static_cast<std::uint8_t>(SourceMask::kFused);
          prior_weight_sum += precision_prior / total;
          ++stats.fused;
        } else {
          // Occlusion, a new front surface or motion: the history is dropped, never blended.
          result.depth_m[i] = static_cast<float>(depth_current);
          result.variance_m2[i] = static_cast<float>(variance_current);
          result.source_mask[i] = static_cast<std::uint8_t>(SourceMask::kCurrent);
          if (depth_current < depth_prior) {
            ++stats.rejected_current_nearer;
          } else {
            ++stats.rejected_current_farther;
          }
        }
      }
      result.valid_mask[i] = 1;
      result.history_age[i] = 0;  // a current measurement supports this pixel
      continue;
    }

    if (prior_present && fusion.fill_holes) {
      const int next_age = prior.age[i] + 1;
      if (next_age <= fusion.max_history_age_frames) {
        // The variance already grew by the process noise during transport (spec 5.4).
        result.depth_m[i] = prior.depth[i];
        result.variance_m2[i] =
            static_cast<float>(std::max<double>(prior.variance[i], floor_variance));
        result.valid_mask[i] = 1;
        result.source_mask[i] = static_cast<std::uint8_t>(SourceMask::kHistoryOnly);
        result.history_age[i] = static_cast<std::uint16_t>(next_age);
        ++stats.history_only;
        continue;
      }
      ++stats.history_expired;
    }
    ++stats.invalid;  // no current measurement and no history we are willing to use
  }

  if (stats.fused > 0) {
    stats.mean_prior_weight = prior_weight_sum / static_cast<double>(stats.fused);
  }
  return stats;
}

}  // namespace cudepthfusion::cpu
