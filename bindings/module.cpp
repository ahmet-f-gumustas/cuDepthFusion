// pybind11 boundary: validates NumPy inputs without hidden copies and returns owned arrays.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "cudepthfusion/build_info.hpp"
#include "cudepthfusion/config.hpp"
#include "cudepthfusion/engine.hpp"
#include "cudepthfusion/error.hpp"
#include "cudepthfusion/types.hpp"

namespace py = pybind11;
namespace cdf = cudepthfusion;

namespace {

std::string describe_array(const py::array& array) {
  std::ostringstream text;
  text << "dtype=" << py::str(array.dtype()).cast<std::string>() << ", shape=(";
  for (py::ssize_t axis = 0; axis < array.ndim(); ++axis) {
    text << (axis > 0 ? ", " : "") << array.shape(axis);
  }
  text << (array.ndim() == 1 ? ",)" : ")");
  return text.str();
}

py::array require_array(const py::handle& object, const std::string& name) {
  if (!py::isinstance<py::array>(object)) {
    throw cdf::InvalidInputError(name + " must be a numpy.ndarray, got " +
                                 py::str(py::type::of(object)).cast<std::string>());
  }
  return py::reinterpret_borrow<py::array>(object);
}

// Rejects layouts that would need a copy; the caller decides whether to copy.
void require_c_contiguous(const py::array& array, const std::string& name) {
  const bool c_contiguous = (array.flags() & py::array::c_style) != 0;
  const bool aligned = array.attr("flags").attr("aligned").cast<bool>();
  if (!c_contiguous || !aligned) {
    throw cdf::InvalidInputError(name + " must be C-contiguous and aligned (" +
                                 describe_array(array) +
                                 "); pass numpy.ascontiguousarray(...) explicitly, no hidden "
                                 "copy is made");
  }
}

cdf::ImageView<float> depth_view(const py::array& depth) {
  if (depth.ndim() != 2) {
    throw cdf::InvalidInputError("depth_m must be 2-D (H, W), got " + describe_array(depth));
  }
  if (!py::isinstance<py::array_t<float>>(depth)) {
    throw cdf::InvalidInputError("depth_m must be float32 metres, got " + describe_array(depth) +
                                 "; convert units in the dataset adapter");
  }
  require_c_contiguous(depth, "depth_m");
  const py::ssize_t height = depth.shape(0);
  const py::ssize_t width = depth.shape(1);
  if (height < 1 || width < 1 || height > cdf::kMaxImageDimension ||
      width > cdf::kMaxImageDimension) {
    std::ostringstream message;
    message << "depth_m shape (" << height << ", " << width << ") is outside [1, "
            << cdf::kMaxImageDimension << "] per dimension";
    throw cdf::InvalidInputError(message.str());
  }
  return {static_cast<const float*>(depth.data()), static_cast<int>(width),
          static_cast<int>(height)};
}

std::optional<cdf::RigidTransform> pose_from(const py::object& object) {
  if (object.is_none()) {
    return std::nullopt;
  }
  const py::array pose = require_array(object, "T_world_camera");
  if (pose.ndim() != 2 || pose.shape(0) != 4 || pose.shape(1) != 4) {
    throw cdf::InvalidInputError("T_world_camera must have shape (4, 4), got " +
                                 describe_array(pose));
  }
  if (!py::isinstance<py::array_t<double>>(pose)) {
    throw cdf::InvalidInputError("T_world_camera must be float64, got " + describe_array(pose));
  }
  require_c_contiguous(pose, "T_world_camera");
  cdf::RigidTransform transform;
  const auto* values = static_cast<const double*>(pose.data());
  std::copy(values, values + transform.m.size(), transform.m.begin());
  return transform;
}

// Hands the vector's buffer to NumPy without copying; the capsule frees it with the array.
template <typename T>
py::array_t<T> to_numpy(std::vector<T>&& values, int height, int width) {
  auto owner = std::make_unique<std::vector<T>>(std::move(values));
  T* data = owner->data();
  py::capsule capsule(owner.get(),
                      [](void* pointer) { delete static_cast<std::vector<T>*>(pointer); });
  owner.release();
  return py::array_t<T>({static_cast<py::ssize_t>(height), static_cast<py::ssize_t>(width)}, data,
                        capsule);
}

py::dict diagnostics_to_dict(const cdf::Diagnostics& diagnostics) {
  py::dict input;
  input["num_pixels"] = diagnostics.input.num_pixels;
  input["num_valid"] = diagnostics.input.num_valid;
  input["num_zero"] = diagnostics.input.num_zero;
  input["num_nonfinite"] = diagnostics.input.num_nonfinite;
  input["num_below_min"] = diagnostics.input.num_below_min;
  input["num_above_max"] = diagnostics.input.num_above_max;

  py::dict out;
  out["frame_index"] = diagnostics.frame_index;
  out["reset_reason"] = cdf::to_string(diagnostics.reset_reason);
  out["temporal_status"] = cdf::to_string(diagnostics.temporal_status);
  out["spatial_applied"] = diagnostics.spatial_applied;
  out["input"] = input;
  out["host_process_ms"] = diagnostics.host_process_ms;
  out["notes"] = py::cast(diagnostics.notes);
  return out;
}

py::dict result_to_dict(cdf::FusionResult&& result) {
  const int h = result.height;
  const int w = result.width;
  py::dict out;
  out["depth_m"] = to_numpy(std::move(result.depth_m), h, w);
  out["valid_mask"] = to_numpy(std::move(result.valid_mask), h, w);
  out["variance_m2"] = to_numpy(std::move(result.variance_m2), h, w);
  out["confidence_score"] = to_numpy(std::move(result.confidence_score), h, w);
  out["source_mask"] = to_numpy(std::move(result.source_mask), h, w);
  out["history_age"] = to_numpy(std::move(result.history_age), h, w);
  out["diagnostics"] = diagnostics_to_dict(result.diagnostics);
  return out;
}

cdf::Backend parse_backend(const std::string& name) {
  if (name == "cpu") {
    return cdf::Backend::kCpu;
  }
  if (name == "cuda") {
    return cdf::Backend::kCuda;
  }
  throw py::value_error("backend must be 'cpu' or 'cuda', got '" + name + "'");
}

py::dict build_info_to_dict() {
  const cdf::BuildInfo info = cdf::build_info();
  py::list devices;
  for (const cdf::CudaDeviceInfo& device : info.cuda_devices) {
    py::dict entry;
    entry["index"] = device.index;
    entry["name"] = device.name;
    entry["compute_capability"] = std::to_string(device.compute_capability_major) + "." +
                                  std::to_string(device.compute_capability_minor);
    entry["total_memory_bytes"] = device.total_memory_bytes;
    entry["multiprocessor_count"] = device.multiprocessor_count;
    devices.append(entry);
  }
  py::dict out;
  out["version"] = info.version;
  out["build_type"] = info.build_type;
  out["cxx_compiler"] = info.cxx_compiler;
  out["cuda_compiled"] = info.cuda_compiled;
  out["cuda_compiler"] = info.cuda_compiled ? py::object(py::str(info.cuda_compiler)) : py::none();
  out["cuda_architectures"] =
      info.cuda_compiled ? py::object(py::str(info.cuda_architectures)) : py::none();
  out["cuda_runtime_version"] =
      info.cuda_compiled ? py::object(py::int_(info.cuda_runtime_version)) : py::none();
  out["cuda_driver_version"] =
      info.cuda_compiled ? py::object(py::int_(info.cuda_driver_version)) : py::none();
  out["cuda_devices"] = devices;
  out["cuda_error"] = info.cuda_error.empty() ? py::object(py::none()) : py::str(info.cuda_error);
  return out;
}

void bind_config(py::module_& m) {
  py::class_<cdf::DepthRangeConfig>(m, "DepthRangeConfig")
      .def(py::init<>())
      .def_readwrite("min_m", &cdf::DepthRangeConfig::min_m)
      .def_readwrite("max_m", &cdf::DepthRangeConfig::max_m);
  py::class_<cdf::SpatialConfig>(m, "SpatialConfig")
      .def(py::init<>())
      .def_readwrite("enabled", &cdf::SpatialConfig::enabled)
      .def_readwrite("radius", &cdf::SpatialConfig::radius)
      .def_readwrite("sigma_xy_px", &cdf::SpatialConfig::sigma_xy_px)
      .def_readwrite("sigma_depth_m", &cdf::SpatialConfig::sigma_depth_m);
  py::class_<cdf::NoiseConfig>(m, "NoiseConfig")
      .def(py::init<>())
      .def_readwrite("a_m", &cdf::NoiseConfig::a_m)
      .def_readwrite("b_per_m", &cdf::NoiseConfig::b_per_m);
  py::class_<cdf::FusionConfig>(m, "FusionConfig")
      .def(py::init<>())
      .def_readwrite("tau_abs_m", &cdf::FusionConfig::tau_abs_m)
      .def_readwrite("k_sigma", &cdf::FusionConfig::k_sigma)
      .def_readwrite("history_decay", &cdf::FusionConfig::history_decay)
      .def_readwrite("max_history_ratio", &cdf::FusionConfig::max_history_ratio)
      .def_readwrite("variance_floor_m2", &cdf::FusionConfig::variance_floor_m2)
      .def_readwrite("variance_reference_m2", &cdf::FusionConfig::variance_reference_m2)
      .def_readwrite("q0_m2", &cdf::FusionConfig::q0_m2)
      .def_readwrite("q_translation", &cdf::FusionConfig::q_translation)
      .def_readwrite("q_rotation_m2_per_rad2", &cdf::FusionConfig::q_rotation_m2_per_rad2)
      .def_readwrite("fill_holes", &cdf::FusionConfig::fill_holes)
      .def_readwrite("max_history_age_frames", &cdf::FusionConfig::max_history_age_frames)
      .def_readwrite("assume_static_camera", &cdf::FusionConfig::assume_static_camera);
  py::class_<cdf::ResetConfig>(m, "ResetConfig")
      .def(py::init<>())
      .def_readwrite("max_frame_gap_s", &cdf::ResetConfig::max_frame_gap_s);
  py::class_<cdf::Config>(m, "Config")
      .def(py::init<>())
      .def_readwrite("depth", &cdf::Config::depth)
      .def_readwrite("spatial", &cdf::Config::spatial)
      .def_readwrite("noise", &cdf::Config::noise)
      .def_readwrite("fusion", &cdf::Config::fusion)
      .def_readwrite("reset", &cdf::Config::reset);
  m.def("validate_config", &cdf::validate_config, py::arg("config"));
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "cuDepthFusion native core";

  py::register_exception<cdf::ConfigError>(m, "ConfigError", PyExc_ValueError);
  py::register_exception<cdf::InvalidInputError>(m, "InvalidInputError", PyExc_ValueError);
  py::register_exception<cdf::BackendUnavailableError>(m, "BackendUnavailableError",
                                                       PyExc_RuntimeError);

  bind_config(m);

  py::class_<cdf::Intrinsics>(m, "Intrinsics")
      .def(py::init([](double fx, double fy, double cx, double cy) {
             return cdf::Intrinsics{fx, fy, cx, cy};
           }),
           py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"))
      .def_readonly("fx", &cdf::Intrinsics::fx)
      .def_readonly("fy", &cdf::Intrinsics::fy)
      .def_readonly("cx", &cdf::Intrinsics::cx)
      .def_readonly("cy", &cdf::Intrinsics::cy)
      .def("__eq__", [](const cdf::Intrinsics& a, const cdf::Intrinsics& b) { return a == b; })
      .def("__repr__", [](const cdf::Intrinsics& k) {
        std::ostringstream text;
        text << "Intrinsics(fx=" << k.fx << ", fy=" << k.fy << ", cx=" << k.cx << ", cy=" << k.cy
             << ")";
        return text.str();
      });

  py::class_<cdf::DepthFusion>(m, "DepthFusion")
      .def(py::init([](const cdf::Config& config, const std::string& backend) {
             return std::make_unique<cdf::DepthFusion>(config, parse_backend(backend));
           }),
           py::arg("config"), py::arg("backend"))
      .def(
          "process",
          [](cdf::DepthFusion& engine, const py::object& depth_m, const cdf::Intrinsics& intrinsics,
             const py::object& T_world_camera, double timestamp_s) {
            const py::array depth = require_array(depth_m, "depth_m");
            cdf::FrameInput frame;
            frame.depth_m = depth_view(depth);
            frame.intrinsics = intrinsics;
            frame.T_world_camera = pose_from(T_world_camera);
            frame.timestamp_s = timestamp_s;
            cdf::FusionResult result;
            {
              // `depth` keeps the input buffer alive; the engine serialises on its own mutex.
              py::gil_scoped_release release;
              result = engine.process(frame);
            }
            return result_to_dict(std::move(result));
          },
          py::kw_only(), py::arg("depth_m"), py::arg("intrinsics"), py::arg("T_world_camera"),
          py::arg("timestamp_s"))
      .def("reset", &cdf::DepthFusion::reset, py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("backend",
                             [](const cdf::DepthFusion& engine) {
                               return std::string(cdf::to_string(engine.backend()));
                             })
      .def_property_readonly("config", [](const cdf::DepthFusion& engine) {
        return engine.config();  // copy: the engine's config is immutable
      });

  m.def("library_version", &cdf::library_version);
  m.def("cuda_compiled", &cdf::cuda_compiled);
  m.def("build_info", &build_info_to_dict,
        "Build and CUDA device information; queries the CUDA runtime when compiled in.");
}
