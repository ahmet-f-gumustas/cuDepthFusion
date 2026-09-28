// CUDA backend: the same algorithm as the CPU reference, one thread per pixel, five kernels
// sequenced in one stream. Buffers are allocated on the first frame and on resize only, never
// inside the steady frame loop (spec 7).
//
// Where the arithmetic is done in double here, it is because the CPU reference does it there
// too and the decision must not depend on the backend: input classification, the measurement
// noise model, the compatibility gate and the merge weights. The bilateral filter runs in
// float (spec 7.2), so filtered depths can differ from the CPU by about 1e-7 m, which can flip
// a pixel that sits exactly on the gate. Those pixels are counted by the parity tests.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "cpu/reproject.hpp"
#include "cuda/device_buffer.hpp"
#include "cuda/device_info.hpp"
#include "cudepthfusion/error.hpp"
#include "pipeline.hpp"

namespace cudepthfusion::detail {
namespace {

using cudepthfusion::cuda::DeviceBuffer;
using cudepthfusion::cuda::Stream;

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;
constexpr unsigned long long kNoWinner = ~0ull;

struct DevicePreviousCamera {
  float inv_fx, inv_fy, cx, cy;
};

struct DeviceCurrentCamera {
  float fx, fy, cx, cy;
};

struct DevicePose {
  float rotation[9];
  float translation[3];
};

struct DeviceConfig {
  double min_m, max_m, noise_a, noise_b;
  int radius;
  float spatial_scale, range_scale;
  double tau_abs, k_sigma, history_decay, max_history_ratio;
  double variance_floor, variance_reference, q_gradient, process_noise, fixed_prior_weight;
  int fill_holes, max_history_age;
};

struct DeviceCounters {
  unsigned long long input_valid, input_zero, input_nonfinite, input_below, input_above;
  unsigned long long candidates, behind_camera, off_screen, visible;
  unsigned long long fused, current_only, rejected_nearer, rejected_farther;
  unsigned long long history_only, history_expired, invalid;
  double prior_weight_sum;
};

__global__ void sanitize_kernel(const float* __restrict__ input, int width, int height,
                                DeviceConfig config, float* __restrict__ depth,
                                std::uint8_t* __restrict__ valid, DeviceCounters* counters) {
  __shared__ unsigned long long tally[5];
  const int thread = threadIdx.y * blockDim.x + threadIdx.x;
  if (thread < 5) {
    tally[thread] = 0ull;
  }
  __syncthreads();

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    const int index = y * width + x;
    const float z = input[index];
    float out_depth = 0.0f;
    unsigned char out_valid = 0;
    if (!isfinite(z)) {
      atomicAdd(&tally[2], 1ull);
    } else if (z == 0.0f) {
      atomicAdd(&tally[1], 1ull);
    } else if (static_cast<double>(z) < config.min_m) {
      atomicAdd(&tally[3], 1ull);
    } else if (static_cast<double>(z) > config.max_m) {
      atomicAdd(&tally[4], 1ull);
    } else {
      out_depth = z;
      out_valid = 1;
      atomicAdd(&tally[0], 1ull);
    }
    depth[index] = out_depth;
    valid[index] = out_valid;
  }
  __syncthreads();
  if (thread == 0) {
    atomicAdd(&counters->input_valid, tally[0]);
    atomicAdd(&counters->input_zero, tally[1]);
    atomicAdd(&counters->input_nonfinite, tally[2]);
    atomicAdd(&counters->input_below, tally[3]);
    atomicAdd(&counters->input_above, tally[4]);
  }
}

__global__ void bilateral_kernel(const float* __restrict__ depth,
                                 const std::uint8_t* __restrict__ valid, int width, int height,
                                 DeviceConfig config, float* __restrict__ filtered) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const int index = y * width + x;
  if (valid[index] == 0) {
    filtered[index] = 0.0f;
    return;
  }
  const float center = depth[index];
  float weight_sum = 0.0f;
  const int y_first = max(y - config.radius, 0);
  const int y_last = min(y + config.radius, height - 1);
  const int x_first = max(x - config.radius, 0);
  const int x_last = min(x + config.radius, width - 1);
  float value_sum = 0.0f;
  for (int ny = y_first; ny <= y_last; ++ny) {
    for (int nx = x_first; nx <= x_last; ++nx) {
      const int neighbour = ny * width + nx;
      if (valid[neighbour] == 0) {
        continue;
      }
      const float dx = static_cast<float>(nx - x);
      const float dy = static_cast<float>(ny - y);
      const float dz = depth[neighbour] - center;
      const float weight =
          expf(-(dx * dx + dy * dy) * config.spatial_scale - dz * dz * config.range_scale);
      weight_sum += weight;
      value_sum += weight * depth[neighbour];
    }
  }
  filtered[index] = value_sum / weight_sum;  // the centre always contributes
}

__global__ void variance_kernel(const float* __restrict__ depth,
                                const std::uint8_t* __restrict__ valid, int count,
                                DeviceConfig config, float* __restrict__ variance) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  if (valid[index] == 0) {
    variance[index] = 0.0f;
    return;
  }
  const double z = depth[index];
  const double sigma = config.noise_a + config.noise_b * z * z;
  variance[index] = static_cast<float>(sigma * sigma);
}

__global__ void project_kernel(const float* __restrict__ history_depth,
                               const std::uint8_t* __restrict__ history_valid, int width,
                               int height, DevicePreviousCamera previous,
                               DeviceCurrentCamera current, DevicePose pose,
                               unsigned long long* __restrict__ winner, DeviceCounters* counters) {
  __shared__ unsigned long long tally[3];
  const int thread = threadIdx.y * blockDim.x + threadIdx.x;
  if (thread < 3) {
    tally[thread] = 0ull;
  }
  __syncthreads();

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    const int source = y * width + x;
    const float z = history_depth[source];
    if (history_valid[source] != 0 && z > 0.0f && isfinite(z)) {
      atomicAdd(&tally[0], 1ull);
      const float ray_x = (static_cast<float>(x) - previous.cx) * previous.inv_fx;
      const float ray_y = (static_cast<float>(y) - previous.cy) * previous.inv_fy;
      const float px = z * ray_x;
      const float py = z * ray_y;
      const float xc = pose.rotation[0] * px + pose.rotation[1] * py + pose.rotation[2] * z +
                       pose.translation[0];
      const float yc = pose.rotation[3] * px + pose.rotation[4] * py + pose.rotation[5] * z +
                       pose.translation[1];
      const float zc = pose.rotation[6] * px + pose.rotation[7] * py + pose.rotation[8] * z +
                       pose.translation[2];
      if (!isfinite(zc) || !(zc > 0.0f)) {
        atomicAdd(&tally[1], 1ull);
      } else {
        const float u = current.fx * (xc / zc) + current.cx;
        const float v = current.fy * (yc / zc) + current.cy;
        if (!isfinite(u) || !isfinite(v) || u < -1.0f || v < -1.0f ||
            u > static_cast<float>(width) || v > static_cast<float>(height)) {
          atomicAdd(&tally[2], 1ull);
        } else {
          const int target_x = static_cast<int>(floorf(u + 0.5f));
          const int target_y = static_cast<int>(floorf(v + 0.5f));
          if (target_x < 0 || target_x >= width || target_y < 0 || target_y >= height) {
            atomicAdd(&tally[2], 1ull);
          } else {
            const unsigned long long key =
                (static_cast<unsigned long long>(__float_as_uint(zc)) << 32) |
                static_cast<unsigned long long>(static_cast<unsigned int>(source));
            atomicMin(&winner[target_y * width + target_x], key);
          }
        }
      }
    }
  }
  __syncthreads();
  if (thread == 0) {
    atomicAdd(&counters->candidates, tally[0]);
    atomicAdd(&counters->behind_camera, tally[1]);
    atomicAdd(&counters->off_screen, tally[2]);
  }
}

__global__ void gather_kernel(const unsigned long long* __restrict__ winner,
                              const float* __restrict__ history_variance,
                              const std::uint16_t* __restrict__ history_age, int width, int height,
                              DevicePreviousCamera previous, DevicePose pose, DeviceConfig config,
                              float* __restrict__ prior_depth, float* __restrict__ prior_variance,
                              std::uint16_t* __restrict__ prior_age,
                              std::uint8_t* __restrict__ prior_present, DeviceCounters* counters) {
  __shared__ unsigned long long visible;
  const int thread = threadIdx.y * blockDim.x + threadIdx.x;
  if (thread == 0) {
    visible = 0ull;
  }
  __syncthreads();

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    const int target = y * width + x;
    const unsigned long long key = winner[target];
    if (key == kNoWinner) {
      prior_depth[target] = 0.0f;
      prior_variance[target] = 0.0f;
      prior_age[target] = 0;
      prior_present[target] = 0;
    } else {
      const unsigned int source = static_cast<unsigned int>(key & 0xFFFFFFFFull);
      const float z = __uint_as_float(static_cast<unsigned int>(key >> 32));
      const int source_x = static_cast<int>(source % static_cast<unsigned int>(width));
      const int source_y = static_cast<int>(source / static_cast<unsigned int>(width));
      const float ray_x = (static_cast<float>(source_x) - previous.cx) * previous.inv_fx;
      const float ray_y = (static_cast<float>(source_y) - previous.cy) * previous.inv_fy;
      const float jacobian = pose.rotation[6] * ray_x + pose.rotation[7] * ray_y + pose.rotation[8];
      const double transported =
          static_cast<double>(jacobian) * jacobian * static_cast<double>(history_variance[source]) +
          config.process_noise;
      prior_depth[target] = z;
      prior_variance[target] = static_cast<float>(transported);
      prior_age[target] = history_age[source];
      prior_present[target] = 1;
      atomicAdd(&visible, 1ull);
    }
  }
  __syncthreads();
  if (thread == 0) {
    atomicAdd(&counters->visible, visible);
  }
}

__device__ inline float local_gradient(const float* depth, const std::uint8_t* valid, int width,
                                       int height, int x, int y, int index) {
  float largest = 0.0f;
  const int offset_x[4] = {-1, 1, 0, 0};
  const int offset_y[4] = {0, 0, -1, 1};
  for (int k = 0; k < 4; ++k) {
    const int nx = x + offset_x[k];
    const int ny = y + offset_y[k];
    if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
      continue;
    }
    const int neighbour = ny * width + nx;
    if (valid[neighbour] == 0) {
      continue;
    }
    largest = fmaxf(largest, fabsf(depth[neighbour] - depth[index]));
  }
  return largest;
}

__global__ void fuse_kernel(const float* __restrict__ depth, const std::uint8_t* __restrict__ valid,
                            const float* __restrict__ variance,
                            const float* __restrict__ prior_depth,
                            const float* __restrict__ prior_variance,
                            const std::uint16_t* __restrict__ prior_age,
                            const std::uint8_t* __restrict__ prior_present, int width, int height,
                            DeviceConfig config, float* __restrict__ out_depth,
                            std::uint8_t* __restrict__ out_valid, float* __restrict__ out_variance,
                            float* __restrict__ out_confidence,
                            std::uint8_t* __restrict__ out_source,
                            std::uint16_t* __restrict__ out_age, DeviceCounters* counters) {
  __shared__ unsigned long long tally[7];
  __shared__ double weight_sum;
  const int thread = threadIdx.y * blockDim.x + threadIdx.x;
  if (thread < 7) {
    tally[thread] = 0ull;
  }
  if (thread == 0) {
    weight_sum = 0.0;
  }
  __syncthreads();

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    const int index = y * width + x;
    float result_depth = 0.0f;
    float result_variance = 0.0f;
    unsigned char result_valid = 0;
    unsigned char result_source = 0;
    unsigned short result_age = 0;

    const bool current_valid = valid[index] != 0;
    const bool prior_here = prior_present[index] != 0;
    if (current_valid) {
      const double depth_current = depth[index];
      const double variance_current =
          fmax(static_cast<double>(variance[index]), config.variance_floor);
      result_valid = 1;
      if (!prior_here) {
        result_depth = static_cast<float>(depth_current);
        result_variance = static_cast<float>(variance_current);
        result_source = 1;
        atomicAdd(&tally[1], 1ull);
      } else {
        const double depth_prior = prior_depth[index];
        const double variance_prior =
            fmax(static_cast<double>(prior_variance[index]), config.variance_floor);
        const double slope = local_gradient(depth, valid, width, height, x, y, index);
        const double variance_merge =
            fmax(static_cast<double>(prior_variance[index]) + config.q_gradient * slope * slope,
                 config.variance_floor);
        const double tau =
            config.tau_abs + config.k_sigma * sqrt(variance_current + variance_prior);
        if (fabs(depth_current - depth_prior) <= tau) {
          const double precision_current = 1.0 / variance_current;
          const double precision_prior = config.fixed_prior_weight > 0.0
                                             ? precision_current * config.fixed_prior_weight /
                                                   (1.0 - config.fixed_prior_weight)
                                             : fmin(config.history_decay / variance_merge,
                                                    config.max_history_ratio * precision_current);
          const double total = precision_current + precision_prior;
          result_depth = static_cast<float>(
              (precision_current * depth_current + precision_prior * depth_prior) / total);
          result_variance = static_cast<float>(fmax(1.0 / total, config.variance_floor));
          result_source = 2;
          atomicAdd(&tally[0], 1ull);
          atomicAdd(&weight_sum, precision_prior / total);
        } else {
          result_depth = static_cast<float>(depth_current);
          result_variance = static_cast<float>(variance_current);
          result_source = 1;
          atomicAdd(&tally[depth_current < depth_prior ? 2 : 3], 1ull);
        }
      }
    } else if (prior_here && config.fill_holes != 0) {
      const int next_age = static_cast<int>(prior_age[index]) + 1;
      if (next_age <= config.max_history_age) {
        result_depth = prior_depth[index];
        result_variance = static_cast<float>(
            fmax(static_cast<double>(prior_variance[index]), config.variance_floor));
        result_valid = 1;
        result_source = 3;
        result_age = static_cast<unsigned short>(next_age);
        atomicAdd(&tally[4], 1ull);
      } else {
        atomicAdd(&tally[5], 1ull);
        atomicAdd(&tally[6], 1ull);
      }
    } else {
      atomicAdd(&tally[6], 1ull);
    }

    out_depth[index] = result_depth;
    out_valid[index] = result_valid;
    out_variance[index] = result_variance;
    out_source[index] = result_source;
    out_age[index] = result_age;
    out_confidence[index] =
        result_valid != 0 ? static_cast<float>(1.0 / (1.0 + static_cast<double>(result_variance) /
                                                                config.variance_reference))
                          : 0.0f;
  }
  __syncthreads();
  if (thread == 0) {
    atomicAdd(&counters->fused, tally[0]);
    atomicAdd(&counters->current_only, tally[1]);
    atomicAdd(&counters->rejected_nearer, tally[2]);
    atomicAdd(&counters->rejected_farther, tally[3]);
    atomicAdd(&counters->history_only, tally[4]);
    atomicAdd(&counters->history_expired, tally[5]);
    atomicAdd(&counters->invalid, tally[6]);
    atomicAdd(&counters->prior_weight_sum, weight_sum);
  }
}

DeviceConfig make_device_config(const Config& config) {
  DeviceConfig device{};
  device.min_m = config.depth.min_m;
  device.max_m = config.depth.max_m;
  device.noise_a = config.noise.a_m;
  device.noise_b = config.noise.b_per_m;
  device.radius = config.spatial.radius;
  device.spatial_scale =
      static_cast<float>(1.0 / (2.0 * config.spatial.sigma_xy_px * config.spatial.sigma_xy_px));
  device.range_scale =
      static_cast<float>(1.0 / (2.0 * config.spatial.sigma_depth_m * config.spatial.sigma_depth_m));
  device.tau_abs = config.fusion.tau_abs_m;
  device.k_sigma = config.fusion.k_sigma;
  device.history_decay = config.fusion.history_decay;
  device.max_history_ratio = config.fusion.max_history_ratio;
  device.variance_floor = config.fusion.variance_floor_m2;
  device.variance_reference = config.fusion.variance_reference_m2;
  device.q_gradient = config.fusion.q_gradient;
  device.fixed_prior_weight = config.fusion.fixed_prior_weight;
  device.process_noise = 0.0;
  device.fill_holes = config.fusion.fill_holes ? 1 : 0;
  device.max_history_age = config.fusion.max_history_age_frames;
  return device;
}

class CudaPipeline final : public Pipeline {
 public:
  CudaPipeline() {
    const cuda::DeviceQuery query = cuda::query_devices();
    if (query.devices.empty()) {
      throw BackendUnavailableError(
          "backend 'cuda' requested, but no usable CUDA device was found: " + query.error);
    }
    // Create the context now, so a broken installation fails here and not mid-sequence.
    CUDEPTHFUSION_CUDA_CHECK(cudaFree(nullptr));
  }

  void reset() override { has_history_ = false; }
  bool has_history() const override { return has_history_; }

  FrameOutcome process(const Config& config, const FrameRequest& request,
                       FusionResult& result) override {
    const FrameInput& frame = *request.frame;
    const int width = result.width;
    const int height = result.height;
    const auto count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    ensure_capacity(width, height);

    const cudaStream_t stream = stream_.get();
    const dim3 block(kBlockX, kBlockY);
    const dim3 grid((static_cast<unsigned>(width) + kBlockX - 1) / kBlockX,
                    (static_cast<unsigned>(height) + kBlockY - 1) / kBlockY);
    DeviceConfig device_config = make_device_config(config);
    FrameOutcome outcome;

    counters_.fill_bytes(0, 1, stream);
    input_.upload(frame.depth_m.data, count, stream);
    sanitize_kernel<<<grid, block, 0, stream>>>(input_.data(), width, height, device_config,
                                                depth_.data(), valid_.data(), counters_.data());

    const float* measurement = depth_.data();
    if (config.spatial.enabled && config.spatial.radius > 0) {
      bilateral_kernel<<<grid, block, 0, stream>>>(depth_.data(), valid_.data(), width, height,
                                                   device_config, filtered_.data());
      measurement = filtered_.data();
      outcome.spatial_applied = true;
    }
    const int linear_threads = 256;
    const int linear_blocks = static_cast<int>((count + linear_threads - 1) / linear_threads);
    variance_kernel<<<linear_blocks, linear_threads, 0, stream>>>(
        measurement, valid_.data(), static_cast<int>(count), device_config, variance_.data());

    if (request.use_history) {
      const cpu::RelativePose relative = cpu::relative_pose(request.pose, history_pose_);
      device_config.process_noise =
          config.fusion.q0_m2 +
          config.fusion.q_translation * relative.translation_norm_m * relative.translation_norm_m +
          config.fusion.q_rotation_m2_per_rad2 * relative.rotation_angle_rad *
              relative.rotation_angle_rad;
      DevicePose pose{};
      for (int i = 0; i < 9; ++i) {
        pose.rotation[i] = relative.rotation[static_cast<std::size_t>(i)];
      }
      for (int i = 0; i < 3; ++i) {
        pose.translation[i] = relative.translation[static_cast<std::size_t>(i)];
      }
      const DevicePreviousCamera previous{static_cast<float>(1.0 / history_intrinsics_.fx),
                                          static_cast<float>(1.0 / history_intrinsics_.fy),
                                          static_cast<float>(history_intrinsics_.cx),
                                          static_cast<float>(history_intrinsics_.cy)};
      const DeviceCurrentCamera current{
          static_cast<float>(frame.intrinsics.fx), static_cast<float>(frame.intrinsics.fy),
          static_cast<float>(frame.intrinsics.cx), static_cast<float>(frame.intrinsics.cy)};
      winner_.fill_bytes(0xFF, count, stream);  // 0xFF... is the "no winner" key
      project_kernel<<<grid, block, 0, stream>>>(history_depth_.data(), history_valid_.data(),
                                                 width, height, previous, current, pose,
                                                 winner_.data(), counters_.data());
      gather_kernel<<<grid, block, 0, stream>>>(
          winner_.data(), history_variance_.data(), history_age_.data(), width, height, previous,
          pose, device_config, prior_depth_.data(), prior_variance_.data(), prior_age_.data(),
          prior_present_.data(), counters_.data());
    } else {
      prior_present_.fill_bytes(0, count, stream);
    }

    fuse_kernel<<<grid, block, 0, stream>>>(
        measurement, valid_.data(), variance_.data(), prior_depth_.data(), prior_variance_.data(),
        prior_age_.data(), prior_present_.data(), width, height, device_config, out_depth_.data(),
        out_valid_.data(), out_variance_.data(), out_confidence_.data(), out_source_.data(),
        out_age_.data(), counters_.data());
    CUDEPTHFUSION_CUDA_CHECK(cudaGetLastError());

    if (request.keep_history) {
      history_depth_.copy_from(out_depth_, count, stream);
      history_variance_.copy_from(out_variance_, count, stream);
      history_valid_.copy_from(out_valid_, count, stream);
      history_age_.copy_from(out_age_, count, stream);
      history_intrinsics_ = frame.intrinsics;
      history_pose_ = request.pose;
      has_history_ = true;
    } else {
      has_history_ = false;
    }

    result.depth_m.resize(count);
    result.valid_mask.resize(count);
    result.variance_m2.resize(count);
    result.confidence_score.resize(count);
    result.source_mask.resize(count);
    result.history_age.resize(count);
    out_depth_.download(result.depth_m.data(), count, stream);
    out_valid_.download(result.valid_mask.data(), count, stream);
    out_variance_.download(result.variance_m2.data(), count, stream);
    out_confidence_.download(result.confidence_score.data(), count, stream);
    out_source_.download(result.source_mask.data(), count, stream);
    out_age_.download(result.history_age.data(), count, stream);
    DeviceCounters host{};
    counters_.download(&host, 1, stream);
    stream_.synchronize();

    outcome.input.num_pixels = count;
    outcome.input.num_valid = host.input_valid;
    outcome.input.num_zero = host.input_zero;
    outcome.input.num_nonfinite = host.input_nonfinite;
    outcome.input.num_below_min = host.input_below;
    outcome.input.num_above_max = host.input_above;
    outcome.fusion.prior_candidates = host.candidates;
    outcome.fusion.prior_behind_camera = host.behind_camera;
    outcome.fusion.prior_off_screen = host.off_screen;
    outcome.fusion.prior_visible = host.visible;
    outcome.fusion.fused = host.fused;
    outcome.fusion.current_only = host.current_only;
    outcome.fusion.rejected_current_nearer = host.rejected_nearer;
    outcome.fusion.rejected_current_farther = host.rejected_farther;
    outcome.fusion.history_only = host.history_only;
    outcome.fusion.history_expired = host.history_expired;
    outcome.fusion.invalid = host.invalid;
    if (host.fused > 0) {
      outcome.fusion.mean_prior_weight = host.prior_weight_sum / static_cast<double>(host.fused);
    }
    return outcome;
  }

 private:
  void ensure_capacity(int width, int height) {
    if (width == width_ && height == height_) {
      return;
    }
    const auto count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    input_.resize(count);
    depth_.resize(count);
    filtered_.resize(count);
    variance_.resize(count);
    valid_.resize(count);
    winner_.resize(count);
    prior_depth_.resize(count);
    prior_variance_.resize(count);
    prior_age_.resize(count);
    prior_present_.resize(count);
    out_depth_.resize(count);
    out_variance_.resize(count);
    out_confidence_.resize(count);
    out_valid_.resize(count);
    out_source_.resize(count);
    out_age_.resize(count);
    history_depth_.resize(count);
    history_variance_.resize(count);
    history_valid_.resize(count);
    history_age_.resize(count);
    counters_.resize(1);
    width_ = width;
    height_ = height;
    has_history_ = false;  // buffers just changed shape
  }

  Stream stream_;
  DeviceBuffer<float> input_, depth_, filtered_, variance_;
  DeviceBuffer<std::uint8_t> valid_;
  DeviceBuffer<unsigned long long> winner_;
  DeviceBuffer<float> prior_depth_, prior_variance_;
  DeviceBuffer<std::uint16_t> prior_age_;
  DeviceBuffer<std::uint8_t> prior_present_;
  DeviceBuffer<float> out_depth_, out_variance_, out_confidence_;
  DeviceBuffer<std::uint8_t> out_valid_, out_source_;
  DeviceBuffer<std::uint16_t> out_age_;
  DeviceBuffer<float> history_depth_, history_variance_;
  DeviceBuffer<std::uint8_t> history_valid_;
  DeviceBuffer<std::uint16_t> history_age_;
  DeviceBuffer<DeviceCounters> counters_;
  Intrinsics history_intrinsics_{};
  RigidTransform history_pose_ = RigidTransform::identity();
  int width_ = 0;
  int height_ = 0;
  bool has_history_ = false;
};

}  // namespace

std::unique_ptr<Pipeline> make_cuda_pipeline() { return std::make_unique<CudaPipeline>(); }

}  // namespace cudepthfusion::detail
