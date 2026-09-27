#pragma once

#include <memory>

#include "cudepthfusion/config.hpp"
#include "cudepthfusion/types.hpp"

namespace cudepthfusion::detail {

// What the engine decided about this frame before any arithmetic happened.
struct FrameRequest {
  const FrameInput* frame = nullptr;
  RigidTransform pose;        // the frame's pose, or identity in static-camera mode
  bool use_history = false;   // run the temporal stage against the stored history
  bool keep_history = false;  // store this result; false also drops what is stored
};

struct FrameOutcome {
  InputStats input;
  FusionStats fusion;
  bool spatial_applied = false;
};

// The per-frame compute path of one backend. The engine keeps the parts that are the same
// everywhere (input validation, reset detection, the temporal decision, diagnostics) and a
// pipeline owns the arithmetic plus the history buffers it lives in.
class Pipeline {
 public:
  virtual ~Pipeline() = default;

  virtual void reset() = 0;              // drop the history
  virtual bool has_history() const = 0;  // is there a history that can be reprojected?

  // Fills every array of `result`; width and height are already set.
  virtual FrameOutcome process(const Config& config, const FrameRequest& request,
                               FusionResult& result) = 0;
};

std::unique_ptr<Pipeline> make_cpu_pipeline();

#ifdef CUDEPTHFUSION_WITH_CUDA
// Throws BackendUnavailableError when no usable CUDA device is present.
std::unique_ptr<Pipeline> make_cuda_pipeline();
#endif

}  // namespace cudepthfusion::detail
