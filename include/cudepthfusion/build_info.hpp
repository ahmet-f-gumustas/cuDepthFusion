#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace cudepthfusion {

struct CudaDeviceInfo {
  int index = 0;
  std::string name;
  int compute_capability_major = 0;
  int compute_capability_minor = 0;
  std::size_t total_memory_bytes = 0;
  int multiprocessor_count = 0;
};

struct BuildInfo {
  std::string version;
  std::string build_type;
  std::string cxx_compiler;
  std::string cuda_compiler;  // empty without CUDA support
  bool cuda_compiled = false;
  std::string cuda_architectures;  // CMAKE_CUDA_ARCHITECTURES used for this build
  int cuda_runtime_version = 0;    // e.g. 12080; 0 when unknown
  int cuda_driver_version = 0;
  std::vector<CudaDeviceInfo> cuda_devices;
  std::string cuda_error;  // why the device query failed, empty on success
};

// Cheap, side-effect free: no CUDA runtime call.
std::string library_version();
bool cuda_compiled();

// Queries the CUDA runtime on every call when CUDA support is compiled in.
BuildInfo build_info();

}  // namespace cudepthfusion
