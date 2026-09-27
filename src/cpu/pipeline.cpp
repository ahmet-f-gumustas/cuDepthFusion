#include <cstdint>
#include <memory>
#include <vector>

#include "cpu/fuse.hpp"
#include "cpu/measurement.hpp"
#include "cpu/reproject.hpp"
#include "cpu/spatial.hpp"
#include "pipeline.hpp"

namespace cudepthfusion::detail {
namespace {

// The reference pipeline: plain C++ on host buffers, the definition of correct behaviour.
class CpuPipeline final : public Pipeline {
 public:
  void reset() override { history_.clear(); }

  bool has_history() const override { return !history_.empty(); }

  FrameOutcome process(const Config& config, const FrameRequest& request,
                       FusionResult& result) override {
    const FrameInput& frame = *request.frame;
    FrameOutcome outcome;

    outcome.input =
        cpu::sanitize_depth(frame.depth_m, config.depth, measurement_.depth, measurement_.valid);
    if (config.spatial.enabled && config.spatial.radius > 0) {
      cpu::bilateral_filter(measurement_.depth, measurement_.valid, result.width, result.height,
                            config.spatial, scratch_);
      measurement_.depth.swap(scratch_);
      outcome.spatial_applied = true;
    }
    cpu::measurement_variance(measurement_.depth, measurement_.valid, config.noise,
                              measurement_.variance);

    prior_.clear();
    cpu::ProjectionStats projection;
    if (request.use_history) {
      const cpu::RelativePose relative = cpu::relative_pose(request.pose, history_.pose);
      projection =
          cpu::gather_prior(history_, frame.intrinsics, relative, config.fusion, winner_, prior_);
    }

    outcome.fusion = cpu::fuse_frame(config, measurement_, prior_, result);
    outcome.fusion.prior_candidates = projection.candidates;
    outcome.fusion.prior_behind_camera = projection.behind_camera;
    outcome.fusion.prior_off_screen = projection.off_screen;
    outcome.fusion.prior_visible = projection.visible;
    cpu::confidence_score(result.variance_m2, result.valid_mask,
                          config.fusion.variance_reference_m2, result.confidence_score);

    if (request.keep_history) {
      history_.depth = result.depth_m;
      history_.variance = result.variance_m2;
      history_.valid = result.valid_mask;
      history_.age = result.history_age;
      history_.intrinsics = frame.intrinsics;
      history_.pose = request.pose;
      history_.width = result.width;
      history_.height = result.height;
    } else {
      history_.clear();  // without a usable pose this frame could never be reprojected
    }
    return outcome;
  }

 private:
  cpu::HistoryState history_;
  cpu::Measurement measurement_;
  cpu::PriorField prior_;
  std::vector<std::uint64_t> winner_;
  std::vector<float> scratch_;
};

}  // namespace

std::unique_ptr<Pipeline> make_cpu_pipeline() { return std::make_unique<CpuPipeline>(); }

}  // namespace cudepthfusion::detail
