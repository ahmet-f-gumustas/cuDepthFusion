#include <cuda_runtime.h>

#include <string>

#include "cuda/device_info.hpp"

namespace cudepthfusion::cuda {
namespace {

std::string describe(const char* call, cudaError_t status) {
  return std::string(call) + " failed: " + cudaGetErrorName(status) + " (" +
         cudaGetErrorString(status) + ")";
}

}  // namespace

DeviceQuery query_devices() {
  DeviceQuery query;
  // Version queries are informational; a failure here is reported by cudaGetDeviceCount.
  cudaRuntimeGetVersion(&query.runtime_version);
  cudaDriverGetVersion(&query.driver_version);

  int count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&count);
  if (count_status != cudaSuccess) {
    query.error = describe("cudaGetDeviceCount", count_status);
    cudaGetLastError();  // reset the runtime's last-error state
    return query;
  }
  if (count == 0) {
    query.error = "no CUDA-capable device detected";
    return query;
  }

  for (int index = 0; index < count; ++index) {
    cudaDeviceProp properties{};
    const cudaError_t status = cudaGetDeviceProperties(&properties, index);
    if (status != cudaSuccess) {
      query.error = describe("cudaGetDeviceProperties", status);
      cudaGetLastError();
      continue;
    }
    CudaDeviceInfo device;
    device.index = index;
    device.name = properties.name;
    device.compute_capability_major = properties.major;
    device.compute_capability_minor = properties.minor;
    device.total_memory_bytes = properties.totalGlobalMem;
    device.multiprocessor_count = properties.multiProcessorCount;
    query.devices.push_back(device);
  }
  return query;
}

}  // namespace cudepthfusion::cuda
