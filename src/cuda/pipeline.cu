// CUDA backend: the same algorithm as the CPU reference, one thread per pixel, six kernels
// sequenced in one stream. Buffers are allocated on the first frame and on resize only, never
// inside the steady frame loop (spec 7); the history is the previous output buffer, swapped in
// place rather than copied.
//
// Precision (spec 7.2): input classification, the measurement noise model and the variance
// transport run in double, as on the CPU, because they are cheap. The bilateral filter, the
// compatibility gate and the merge run in float: the CPU reference keeps double there, and a
// pixel whose depth difference sits within float rounding of tau can fall to the other side.
// The parity tests count those pixels (spec 9.3); P7 measured none on 150 real kt1 frames beyond
// the z-buffer ties that were already there (docs/PERFORMANCE.md).

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "cpu/reproject.hpp"
#include "cuda/device_buffer.hpp"
#include "cuda/device_info.hpp"
#include "cudepthfusion/error.hpp"
#include "pipeline.hpp"

namespace cudepthfusion::detail {
namespace {

using cudepthfusion::cuda::DeviceBuffer;
using cudepthfusion::cuda::Event;
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
  double process_noise;
};

struct DeviceCounters {
  unsigned long long input_valid, input_zero, input_nonfinite, input_below, input_above;
  unsigned long long candidates, behind_camera, off_screen, visible;
  unsigned long long fused, current_only, rejected_nearer, rejected_farther;
  unsigned long long history_only, history_expired, invalid;
  double prior_weight_sum;
};

// Counting happens per warp: a ballot tells lane 0 how many lanes matched, and lane 0 adds that
// to the block's shared tally with one atomic. Every thread of the block must reach these calls,
// including threads outside the image, because the ballot uses the full warp mask.
__device__ inline bool is_warp_leader() {
  return ((threadIdx.y * blockDim.x + threadIdx.x) & 31u) == 0u;
}

__device__ inline void warp_count(bool flag, unsigned long long* tally) {
  const unsigned int lanes = __ballot_sync(0xFFFFFFFFu, flag);
  if (is_warp_leader() && lanes != 0u) {
    atomicAdd(tally, static_cast<unsigned long long>(__popc(lanes)));
  }
}

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
  int category = -1;  // outside the image: counted nowhere
  if (x < width && y < height) {
    const int index = y * width + x;
    const float z = input[index];
    float out_depth = 0.0f;
    unsigned char out_valid = 0;
    if (!isfinite(z)) {
      category = 2;
    } else if (z == 0.0f) {
      category = 1;
    } else if (static_cast<double>(z) < config.min_m) {
      category = 3;
    } else if (static_cast<double>(z) > config.max_m) {
      category = 4;
    } else {
      out_depth = z;
      out_valid = 1;
      category = 0;
    }
    depth[index] = out_depth;
    valid[index] = out_valid;
  }
  for (int k = 0; k < 5; ++k) {
    warp_count(category == k, &tally[k]);
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

// Mask-aware bilateral filter reading its window straight from global memory. A shared-memory
// tile with a halo (spec 7.4) was built and measured in P7: bit-identical output, no speed-up
// (19.1 us global against 20.6 us tiled at 640x480, radius 2, in a run where the untouched
// kernels were ~7% slower too), because the 25 expf per pixel bound the kernel and the L1/L2
// caches already serve the overlapping loads. The simpler version stays.
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
  bool candidate = false;
  bool behind = false;
  bool off_screen = false;
  if (x < width && y < height) {
    const int source = y * width + x;
    const float z = history_depth[source];
    if (history_valid[source] != 0 && z > 0.0f && isfinite(z)) {
      candidate = true;
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
        behind = true;
      } else {
        const float u = current.fx * (xc / zc) + current.cx;
        const float v = current.fy * (yc / zc) + current.cy;
        if (!isfinite(u) || !isfinite(v) || u < -1.0f || v < -1.0f ||
            u > static_cast<float>(width) || v > static_cast<float>(height)) {
          off_screen = true;
        } else {
          const int target_x = static_cast<int>(floorf(u + 0.5f));
          const int target_y = static_cast<int>(floorf(v + 0.5f));
          if (target_x < 0 || target_x >= width || target_y < 0 || target_y >= height) {
            off_screen = true;
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
  warp_count(candidate, &tally[0]);
  warp_count(behind, &tally[1]);
  warp_count(off_screen, &tally[2]);
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
  bool present = false;
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
      present = true;
    }
  }
  warp_count(present, &visible);
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

// The gate and the merge in FP32 (spec 7.2): this kernel runs on the FP64 pipe otherwise, which
// on consumer GPUs is 1/64 of the FP32 rate and made it the most expensive stage by far (P7,
// docs/PERFORMANCE.md). The CPU reference keeps double. The two agree exactly except for a pixel
// whose |current - prior| lies within float rounding of tau, which the parity tests count
// separately (spec 9.3). Constants arrive already narrowed to float.
struct FuseConstants {
  float tau_abs, k_sigma, history_decay, max_history_ratio;
  float variance_floor, variance_reference, q_gradient, fixed_prior_ratio;
  int use_fixed_weight, fill_holes, max_history_age;
};

__global__ void fuse_kernel(const float* __restrict__ depth, const std::uint8_t* __restrict__ valid,
                            const float* __restrict__ variance,
                            const float* __restrict__ prior_depth,
                            const float* __restrict__ prior_variance,
                            const std::uint16_t* __restrict__ prior_age,
                            const std::uint8_t* __restrict__ prior_present, int width, int height,
                            FuseConstants constants, float* __restrict__ out_depth,
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
  int decision = -1;  // index into tally, -1 outside the image
  bool expired = false;
  float prior_weight = 0.0f;
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
      const float depth_current = depth[index];
      const float variance_current = fmaxf(variance[index], constants.variance_floor);
      result_valid = 1;
      if (!prior_here) {
        result_depth = depth_current;
        result_variance = variance_current;
        result_source = 1;
        decision = 1;
      } else {
        const float depth_prior = prior_depth[index];
        const float variance_prior = fmaxf(prior_variance[index], constants.variance_floor);
        const float slope = local_gradient(depth, valid, width, height, x, y, index);
        const float variance_merge = fmaxf(
            prior_variance[index] + constants.q_gradient * slope * slope, constants.variance_floor);
        const float tau =
            constants.tau_abs + constants.k_sigma * sqrtf(variance_current + variance_prior);
        if (fabsf(depth_current - depth_prior) <= tau) {
          const float precision_current = 1.0f / variance_current;
          const float precision_prior =
              constants.use_fixed_weight != 0
                  ? precision_current * constants.fixed_prior_ratio
                  : fminf(constants.history_decay / variance_merge,
                          constants.max_history_ratio * precision_current);
          const float inverse_total = 1.0f / (precision_current + precision_prior);
          const float weight = precision_prior * inverse_total;
          // The weighted mean written as current + w * (prior - current): in float this keeps
          // the error at an ulp of the depth, the difference being small.
          result_depth = depth_current + weight * (depth_prior - depth_current);
          result_variance = fmaxf(inverse_total, constants.variance_floor);
          result_source = 2;
          decision = 0;
          prior_weight = weight;
        } else {
          result_depth = depth_current;
          result_variance = variance_current;
          result_source = 1;
          decision = depth_current < depth_prior ? 2 : 3;
        }
      }
    } else if (prior_here && constants.fill_holes != 0) {
      const int next_age = static_cast<int>(prior_age[index]) + 1;
      if (next_age <= constants.max_history_age) {
        result_depth = prior_depth[index];
        result_variance = fmaxf(prior_variance[index], constants.variance_floor);
        result_valid = 1;
        result_source = 3;
        result_age = static_cast<unsigned short>(next_age);
        decision = 4;
      } else {
        expired = true;
        decision = 6;
      }
    } else {
      decision = 6;
    }

    out_depth[index] = result_depth;
    out_valid[index] = result_valid;
    out_variance[index] = result_variance;
    out_source[index] = result_source;
    out_age[index] = result_age;
    // 1 / (1 + v / ref) written as ref / (ref + v): the same value with one division.
    out_confidence[index] = result_valid != 0 ? constants.variance_reference /
                                                    (constants.variance_reference + result_variance)
                                              : 0.0f;
  }
  for (int k = 0; k < 7; ++k) {
    if (k != 5) {
      warp_count(decision == k, &tally[k]);
    }
  }
  warp_count(expired, &tally[5]);
  // Sum the float weights across the warp, then widen once per warp for the block total.
  float warp_weight = prior_weight;
  for (int offset = 16; offset > 0; offset /= 2) {
    warp_weight += __shfl_down_sync(0xFFFFFFFFu, warp_weight, offset);
  }
  if (is_warp_leader() && warp_weight != 0.0f) {
    atomicAdd(&weight_sum, static_cast<double>(warp_weight));
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

FuseConstants make_fuse_constants(const Config& config) {
  const FusionConfig& fusion = config.fusion;
  FuseConstants constants{};
  constants.tau_abs = static_cast<float>(fusion.tau_abs_m);
  constants.k_sigma = static_cast<float>(fusion.k_sigma);
  constants.history_decay = static_cast<float>(fusion.history_decay);
  constants.max_history_ratio = static_cast<float>(fusion.max_history_ratio);
  constants.variance_floor = static_cast<float>(fusion.variance_floor_m2);
  constants.variance_reference = static_cast<float>(fusion.variance_reference_m2);
  constants.q_gradient = static_cast<float>(fusion.q_gradient);
  constants.use_fixed_weight = fusion.fixed_prior_weight > 0.0 ? 1 : 0;
  constants.fixed_prior_ratio =
      fusion.fixed_prior_weight > 0.0
          ? static_cast<float>(fusion.fixed_prior_weight / (1.0 - fusion.fixed_prior_weight))
          : 0.0f;
  constants.fill_holes = fusion.fill_holes ? 1 : 0;
  constants.max_history_age = fusion.max_history_age_frames;
  return constants;
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
  device.process_noise = 0.0;
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

    // The caller's array is pageable; staging it in pinned memory makes the upload a single
    // full-speed DMA instead of the driver's chunked bounce copy.
    std::memcpy(input_staging_.data(), frame.depth_m.data, count * sizeof(float));
    start_.record(stream);
    input_.upload(reinterpret_cast<const float*>(input_staging_.data()), count, stream);
    uploaded_.record(stream);
    counters_.fill_bytes(0, 1, stream);
    sanitize_kernel<<<grid, block, 0, stream>>>(input_.data(), width, height, device_config,
                                                depth_.data(), valid_.data(), counters_.data());

    sanitized_.record(stream);

    const float* measurement = depth_.data();
    if (config.spatial.enabled && config.spatial.radius > 0) {
      bilateral_kernel<<<grid, block, 0, stream>>>(depth_.data(), valid_.data(), width, height,
                                                   device_config, filtered_.data());
      measurement = filtered_.data();
      outcome.spatial_applied = true;
    }
    filtered_event_.record(stream);
    const int linear_threads = 256;
    const int linear_blocks = static_cast<int>((count + linear_threads - 1) / linear_threads);
    variance_kernel<<<linear_blocks, linear_threads, 0, stream>>>(
        measurement, valid_.data(), static_cast<int>(count), device_config, variance_.data());
    variance_event_.record(stream);

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
    reprojected_.record(stream);

    fuse_kernel<<<grid, block, 0, stream>>>(
        measurement, valid_.data(), variance_.data(), prior_depth_.data(), prior_variance_.data(),
        prior_age_.data(), prior_present_.data(), width, height, make_fuse_constants(config),
        out_depth_.data(), out_valid_.data(), out_variance_.data(), out_confidence_.data(),
        out_source_.data(), out_age_.data(), counters_.data());
    CUDEPTHFUSION_CUDA_CHECK(cudaGetLastError());
    fused_.record(stream);

    // Keeping the result as history costs nothing on the device: once the downloads below are
    // queued, the output and history buffers trade places (ping-pong). The stream orders the
    // downloads before the next frame's kernels overwrite what is then the output buffer.
    history_event_.record(stream);

    // Every output lands in one pinned block, and each is copied into its owned result vector
    // as soon as its own transfer has finished, so the host copies overlap the remaining DMA.
    const StagingLayout layout = staging_layout(count);
    unsigned char* staging = staging_.data();
    out_depth_.download(reinterpret_cast<float*>(staging + layout.depth), count, stream);
    output_ready_[0].record(stream);
    out_variance_.download(reinterpret_cast<float*>(staging + layout.variance), count, stream);
    output_ready_[1].record(stream);
    out_confidence_.download(reinterpret_cast<float*>(staging + layout.confidence), count, stream);
    output_ready_[2].record(stream);
    out_age_.download(reinterpret_cast<std::uint16_t*>(staging + layout.age), count, stream);
    output_ready_[3].record(stream);
    out_valid_.download(staging + layout.valid, count, stream);
    output_ready_[4].record(stream);
    out_source_.download(staging + layout.source, count, stream);
    counters_.download(reinterpret_cast<DeviceCounters*>(staging + layout.counters), 1, stream);
    downloaded_.record(stream);
    if (request.keep_history) {
      std::swap(history_depth_, out_depth_);
      std::swap(history_variance_, out_variance_);
      std::swap(history_valid_, out_valid_);
      std::swap(history_age_, out_age_);
      history_intrinsics_ = frame.intrinsics;
      history_pose_ = request.pose;
      has_history_ = true;
    } else {
      has_history_ = false;
    }

    output_ready_[0].synchronize();
    copy_out(staging + layout.depth, count, result.depth_m);
    output_ready_[1].synchronize();
    copy_out(staging + layout.variance, count, result.variance_m2);
    output_ready_[2].synchronize();
    copy_out(staging + layout.confidence, count, result.confidence_score);
    output_ready_[3].synchronize();
    copy_out(staging + layout.age, count, result.history_age);
    output_ready_[4].synchronize();
    copy_out(staging + layout.valid, count, result.valid_mask);
    stream_.synchronize();
    copy_out(staging + layout.source, count, result.source_mask);
    DeviceCounters host{};
    std::memcpy(&host, staging + layout.counters, sizeof(DeviceCounters));

    outcome.device.measured = true;
    outcome.device.upload_ms = uploaded_.since(start_);
    outcome.device.compute_ms = history_event_.since(uploaded_);
    outcome.device.download_ms = downloaded_.since(history_event_);
    outcome.device.sanitize_ms = sanitized_.since(uploaded_);
    outcome.device.bilateral_ms = filtered_event_.since(sanitized_);
    outcome.device.variance_ms = variance_event_.since(filtered_event_);
    outcome.device.reproject_ms = reprojected_.since(variance_event_);
    outcome.device.fuse_ms = fused_.since(reprojected_);
    outcome.device.history_ms = history_event_.since(fused_);

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
  struct StagingLayout {
    std::size_t depth, variance, confidence, age, valid, source, counters, total;
  };

  // Byte offsets of each output inside the pinned block, every one 256-byte aligned.
  static StagingLayout staging_layout(std::size_t count) {
    const auto align = [](std::size_t bytes) {
      return (bytes + 255u) & ~static_cast<std::size_t>(255u);
    };
    StagingLayout layout{};
    std::size_t offset = 0;
    const auto take = [&](std::size_t bytes) {
      const std::size_t start = offset;
      offset += align(bytes);
      return start;
    };
    layout.depth = take(count * sizeof(float));
    layout.variance = take(count * sizeof(float));
    layout.confidence = take(count * sizeof(float));
    layout.age = take(count * sizeof(std::uint16_t));
    layout.valid = take(count);
    layout.source = take(count);
    layout.counters = take(sizeof(DeviceCounters));
    layout.total = offset;
    return layout;
  }

  // assign() copies straight from the staging block; resize() + memcpy would first zero-fill the
  // vector, one more pass over every output.
  template <typename T>
  static void copy_out(const unsigned char* source, std::size_t count, std::vector<T>& target) {
    const auto* first = reinterpret_cast<const T*>(source);
    target.assign(first, first + count);
  }

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
    staging_.reserve(staging_layout(count).total);
    input_staging_.reserve(count * sizeof(float));
    width_ = width;
    height_ = height;
    has_history_ = false;  // buffers just changed shape
  }

  Stream stream_;
  Event start_, uploaded_, sanitized_, filtered_event_, variance_event_, reprojected_, fused_,
      history_event_, downloaded_;
  Event output_ready_[5];
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
  cuda::PinnedBuffer staging_;
  cuda::PinnedBuffer input_staging_;
  Intrinsics history_intrinsics_{};
  RigidTransform history_pose_ = RigidTransform::identity();
  int width_ = 0;
  int height_ = 0;
  bool has_history_ = false;
};

}  // namespace

std::unique_ptr<Pipeline> make_cuda_pipeline() { return std::make_unique<CudaPipeline>(); }

}  // namespace cudepthfusion::detail
