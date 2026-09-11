#pragma once

#include <memory>

#include "cudepthfusion/config.hpp"
#include "cudepthfusion/types.hpp"

namespace cudepthfusion {

// Largest supported image. The z-buffer key packs the source pixel index into 32 bits.
inline constexpr int kMaxImageDimension = 16384;

// Synchronous temporal depth fusion engine.
//
// One engine processes one frame at a time; concurrent process() calls serialise on an
// internal mutex. Backend::kCuda never falls back to the CPU: if CUDA is unavailable the
// constructor throws BackendUnavailableError.
//
// Moving is not synchronised: never move an engine while another thread is inside
// process() or reset(). A moved-from engine must not be used.
class DepthFusion {
 public:
  DepthFusion(const Config& config, Backend backend);
  ~DepthFusion();

  DepthFusion(const DepthFusion&) = delete;
  DepthFusion& operator=(const DepthFusion&) = delete;
  DepthFusion(DepthFusion&&) noexcept;
  DepthFusion& operator=(DepthFusion&&) noexcept;

  // Throws InvalidInputError when the frame violates the data contract.
  FusionResult process(const FrameInput& frame);

  // Drops history; the next frame reports ResetReason::kExplicit.
  void reset();

  const Config& config() const;
  Backend backend() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace cudepthfusion
