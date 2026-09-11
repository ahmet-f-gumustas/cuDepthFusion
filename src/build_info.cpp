#include "cudepthfusion/build_info.hpp"

#include <utility>

#ifdef CUDEPTHFUSION_WITH_CUDA
#include "cuda/device_info.hpp"
#endif

namespace cudepthfusion {

std::string library_version() { return CUDEPTHFUSION_VERSION; }

bool cuda_compiled() {
#ifdef CUDEPTHFUSION_WITH_CUDA
  return true;
#else
  return false;
#endif
}

BuildInfo build_info() {
  BuildInfo info;
  info.version = library_version();
  info.build_type = CUDEPTHFUSION_BUILD_TYPE;
  info.cxx_compiler = CUDEPTHFUSION_CXX_COMPILER;
  info.cuda_compiled = cuda_compiled();
#ifdef CUDEPTHFUSION_WITH_CUDA
  info.cuda_compiler = CUDEPTHFUSION_CUDA_COMPILER;
  info.cuda_architectures = CUDEPTHFUSION_CUDA_ARCHITECTURES;
  cuda::DeviceQuery query = cuda::query_devices();
  info.cuda_runtime_version = query.runtime_version;
  info.cuda_driver_version = query.driver_version;
  info.cuda_devices = std::move(query.devices);
  info.cuda_error = std::move(query.error);
#endif
  return info;
}

}  // namespace cudepthfusion
