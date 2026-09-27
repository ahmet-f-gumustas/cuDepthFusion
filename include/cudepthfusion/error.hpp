#pragma once

#include <stdexcept>

namespace cudepthfusion {

// Base class for every error the library throws on purpose.
class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// A configuration value is missing, out of range or inconsistent.
class ConfigError : public Error {
 public:
  using Error::Error;
};

// A frame passed to process() violates the data contract (shape, calibration, timestamp).
class InvalidInputError : public Error {
 public:
  using Error::Error;
};

// The requested backend cannot run in this build or on this machine. Never silently replaced.
class BackendUnavailableError : public Error {
 public:
  using Error::Error;
};

// A CUDA call failed. The message carries the CUDA error name and the call site.
class CudaError : public Error {
 public:
  using Error::Error;
};

}  // namespace cudepthfusion
