#pragma once

#include <string>
#include <vector>

#include "cudepthfusion/build_info.hpp"

namespace cudepthfusion::cuda {

struct DeviceQuery {
  std::vector<CudaDeviceInfo> devices;
  int runtime_version = 0;
  int driver_version = 0;
  std::string error;  // empty when the query succeeded
};

DeviceQuery query_devices();

}  // namespace cudepthfusion::cuda
