// CPU/GPU parity (spec 9.3). Skipped unless this build has CUDA and a device.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "cudepthfusion/build_info.hpp"
#include "cudepthfusion/engine.hpp"

namespace cudepthfusion {
namespace {

constexpr int kWidth = 96;
constexpr int kHeight = 72;
constexpr Intrinsics kIntrinsics{90.0, 90.0, 47.5, 35.5};
constexpr double kDepthTolerance = 1e-5;  // metres, the spec's parity tolerance
constexpr double kVarianceRelTolerance = 1e-4;

struct Frame {
  std::vector<float> depth;
  RigidTransform pose;
  double timestamp_s = 0.0;
};

// A slanted plane with a depth step, a square crossing in front of it, and dropouts. The
// camera translates, so the temporal stage has real work to do.
std::vector<Frame> make_sequence(int count, float noise_sigma, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> noise(0.0f, noise_sigma);
  std::vector<Frame> frames;
  frames.reserve(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    Frame frame;
    frame.depth.resize(static_cast<std::size_t>(kWidth * kHeight));
    frame.pose = RigidTransform::identity();
    frame.pose.m[3] = 0.01 * index;  // slide along +x
    frame.timestamp_s = index / 30.0;
    const int square_x = 8 + index;
    for (int y = 0; y < kHeight; ++y) {
      for (int x = 0; x < kWidth; ++x) {
        float z = 2.0f + 0.0015f * (static_cast<float>(x) - 47.5f);
        if (x > 3 * kWidth / 4) {
          z += 1.0f;  // a depth step the gate must not blur
        }
        if (x >= square_x && x < square_x + 12 && y >= 20 && y < 32) {
          z = 1.2f;  // a front surface crossing the view
        }
        if ((x + 3 * y + index) % 101 == 0) {
          z = 0.0f;  // sensor dropout
        }
        const auto i = static_cast<std::size_t>(y * kWidth + x);
        frame.depth[i] = z > 0.0f && noise_sigma > 0.0f ? z + noise(rng) : z;
      }
    }
    frames.push_back(std::move(frame));
  }
  return frames;
}

FrameInput to_input(const Frame& frame) {
  FrameInput input;
  input.depth_m = ImageView<float>{frame.depth.data(), kWidth, kHeight};
  input.intrinsics = kIntrinsics;
  input.T_world_camera = frame.pose;
  input.timestamp_s = frame.timestamp_s;
  return input;
}

bool cuda_available() {
  const BuildInfo info = build_info();
  return info.cuda_compiled && !info.cuda_devices.empty();
}

struct Mismatch {
  std::size_t valid = 0;
  std::size_t source = 0;
  double max_depth_difference = 0.0;
  double max_variance_relative = 0.0;
  std::size_t compared = 0;
};

Mismatch compare(const FusionResult& cpu, const FusionResult& gpu) {
  Mismatch mismatch;
  for (std::size_t i = 0; i < cpu.depth_m.size(); ++i) {
    if (cpu.valid_mask[i] != gpu.valid_mask[i]) {
      ++mismatch.valid;
      continue;
    }
    if (cpu.source_mask[i] != gpu.source_mask[i]) {
      ++mismatch.source;
      continue;
    }
    if (cpu.valid_mask[i] == 0) {
      continue;
    }
    ++mismatch.compared;
    mismatch.max_depth_difference =
        std::max(mismatch.max_depth_difference,
                 std::fabs(static_cast<double>(cpu.depth_m[i]) - gpu.depth_m[i]));
    const double reference = std::max<double>(cpu.variance_m2[i], 1e-12);
    mismatch.max_variance_relative = std::max(
        mismatch.max_variance_relative,
        std::fabs(static_cast<double>(cpu.variance_m2[i]) - gpu.variance_m2[i]) / reference);
  }
  return mismatch;
}

TEST(Parity, NoiseFreeSequenceMatchesBitForBitInTheMasks) {
  if (!cuda_available()) {
    GTEST_SKIP() << "no CUDA build or no device";
  }
  const std::vector<Frame> frames = make_sequence(12, 0.0f, 1);
  DepthFusion cpu(Config{}, Backend::kCpu);
  DepthFusion gpu(Config{}, Backend::kCuda);

  for (const Frame& frame : frames) {
    const FusionResult cpu_result = cpu.process(to_input(frame));
    const FusionResult gpu_result = gpu.process(to_input(frame));
    const Mismatch mismatch = compare(cpu_result, gpu_result);

    // Without noise every pixel sits far from the gate, so the decisions must agree exactly.
    EXPECT_EQ(mismatch.valid, 0u);
    EXPECT_EQ(mismatch.source, 0u);
    EXPECT_LT(mismatch.max_depth_difference, kDepthTolerance);
    EXPECT_LT(mismatch.max_variance_relative, kVarianceRelTolerance);
    EXPECT_EQ(cpu_result.diagnostics.input.num_valid, gpu_result.diagnostics.input.num_valid);
    EXPECT_EQ(cpu_result.diagnostics.fusion.fused, gpu_result.diagnostics.fusion.fused);
    EXPECT_EQ(cpu_result.diagnostics.fusion.prior_visible,
              gpu_result.diagnostics.fusion.prior_visible);
    EXPECT_EQ(cpu_result.diagnostics.temporal_status, gpu_result.diagnostics.temporal_status);
  }
}

TEST(Parity, NoisySequenceAgreesExceptOnPixelsSittingOnTheGate) {
  if (!cuda_available()) {
    GTEST_SKIP() << "no CUDA build or no device";
  }
  const std::vector<Frame> frames = make_sequence(12, 0.006f, 7);
  DepthFusion cpu(Config{}, Backend::kCpu);
  DepthFusion gpu(Config{}, Backend::kCuda);

  std::size_t source_mismatches = 0;
  std::size_t pixels = 0;
  double worst_depth = 0.0;
  for (const Frame& frame : frames) {
    const FusionResult cpu_result = cpu.process(to_input(frame));
    const FusionResult gpu_result = gpu.process(to_input(frame));
    const Mismatch mismatch = compare(cpu_result, gpu_result);
    EXPECT_EQ(mismatch.valid, 0u);  // validity never depends on float rounding
    source_mismatches += mismatch.source;
    pixels += cpu_result.depth_m.size();
    worst_depth = std::max(worst_depth, mismatch.max_depth_difference);
  }
  // The bilateral filter runs in float on the GPU (spec 7.2), so a pixel whose depth
  // difference sits exactly on tau can fall to the other side. Report the share.
  const double share = static_cast<double>(source_mismatches) / static_cast<double>(pixels);
  EXPECT_LT(share, 0.005) << source_mismatches << " of " << pixels << " pixels";
  EXPECT_LT(worst_depth, kDepthTolerance);
  std::cout << "  near-gate source differences: " << source_mismatches << " of " << pixels << " ("
            << 100.0 * share << " %)\n";
}

TEST(Parity, ResetPoseLossAndResizeBehaveTheSame) {
  if (!cuda_available()) {
    GTEST_SKIP() << "no CUDA build or no device";
  }
  const std::vector<Frame> frames = make_sequence(6, 0.0f, 3);
  DepthFusion cpu(Config{}, Backend::kCpu);
  DepthFusion gpu(Config{}, Backend::kCuda);

  for (std::size_t index = 0; index < frames.size(); ++index) {
    FrameInput input = to_input(frames[index]);
    if (index == 2) {
      input.T_world_camera.reset();  // pose loss must drop the history on both backends
    }
    if (index == 4) {
      cpu.reset();
      gpu.reset();
    }
    const FusionResult cpu_result = cpu.process(input);
    const FusionResult gpu_result = gpu.process(input);
    EXPECT_EQ(cpu_result.diagnostics.reset_reason, gpu_result.diagnostics.reset_reason);
    EXPECT_EQ(cpu_result.diagnostics.temporal_status, gpu_result.diagnostics.temporal_status);
    EXPECT_EQ(compare(cpu_result, gpu_result).source, 0u);
  }
}

}  // namespace
}  // namespace cudepthfusion
