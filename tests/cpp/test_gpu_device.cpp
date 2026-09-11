#include <gtest/gtest.h>

#include "cudepthfusion/build_info.hpp"

namespace cudepthfusion {
namespace {

TEST(GpuDevice, QueryReportsAtLeastOneUsableDevice) {
  const BuildInfo info = build_info();
  if (!info.cuda_compiled) {
    GTEST_SKIP() << "built without CUDA support";
  }
  if (info.cuda_devices.empty()) {
    GTEST_SKIP() << "no CUDA device: " << info.cuda_error;
  }
  EXPECT_GT(info.cuda_runtime_version, 0);
  EXPECT_GT(info.cuda_driver_version, 0);
  EXPECT_FALSE(info.cuda_architectures.empty());
  for (const CudaDeviceInfo& device : info.cuda_devices) {
    EXPECT_FALSE(device.name.empty());
    EXPECT_GE(device.compute_capability_major, 5);
    EXPECT_GT(device.total_memory_bytes, 0u);
    EXPECT_GT(device.multiprocessor_count, 0);
  }
}

}  // namespace
}  // namespace cudepthfusion
