// Device memory must settle: no growth over a long run, across resets and resizes (spec 9.3).

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "cudepthfusion/build_info.hpp"
#include "cudepthfusion/engine.hpp"

namespace cudepthfusion {
namespace {

std::size_t free_device_memory() {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
    return 0;
  }
  return free_bytes;
}

FusionResult run_frame(DepthFusion& engine, std::vector<float>& depth, int width, int height,
                       double timestamp_s) {
  FrameInput input;
  input.depth_m = ImageView<float>{depth.data(), width, height};
  input.intrinsics = Intrinsics{90.0, 90.0, width * 0.5 - 0.5, height * 0.5 - 0.5};
  input.T_world_camera = RigidTransform::identity();
  input.timestamp_s = timestamp_s;
  return engine.process(input);
}

TEST(CudaMemory, SettlesAcrossManyFramesResetsAndResizes) {
  const BuildInfo info = build_info();
  if (!info.cuda_compiled || info.cuda_devices.empty()) {
    GTEST_SKIP() << "no CUDA build or no device";
  }

  DepthFusion engine(Config{}, Backend::kCuda);
  std::vector<float> small(64 * 48, 2.0f);
  std::vector<float> large(128 * 96, 2.0f);

  for (int i = 0; i < 20; ++i) {  // warm-up: this is where the buffers are allocated
    run_frame(engine, small, 64, 48, i / 30.0);
  }
  const std::size_t after_warmup = free_device_memory();
  ASSERT_GT(after_warmup, 0u);

  for (int i = 0; i < 200; ++i) {
    run_frame(engine, small, 64, 48, (20 + i) / 30.0);
    if (i % 50 == 0) {
      engine.reset();
    }
  }
  const std::size_t after_long_run = free_device_memory();
  EXPECT_EQ(after_long_run, after_warmup) << "the steady frame loop must not allocate";

  // A larger frame grows the buffers once; going back to the small one must not grow again.
  for (int i = 0; i < 5; ++i) {
    run_frame(engine, large, 128, 96, (300 + i) / 30.0);
  }
  const std::size_t after_resize = free_device_memory();
  for (int i = 0; i < 50; ++i) {
    run_frame(engine, large, 128, 96, (400 + i) / 30.0);
    run_frame(engine, small, 64, 48, (400 + i) / 30.0 + 0.001);
  }
  EXPECT_EQ(free_device_memory(), after_resize);
}

}  // namespace
}  // namespace cudepthfusion
