#include "cudepthfusion/geometry.hpp"

#include <cmath>
#include <sstream>

#include "cudepthfusion/error.hpp"

namespace cudepthfusion {

void validate_intrinsics(const Intrinsics& intrinsics) {
  const bool all_finite = std::isfinite(intrinsics.fx) && std::isfinite(intrinsics.fy) &&
                          std::isfinite(intrinsics.cx) && std::isfinite(intrinsics.cy);
  if (!all_finite || !(intrinsics.fx > 0.0) || intrinsics.fy == 0.0) {
    std::ostringstream message;
    message << "invalid intrinsics (fx=" << intrinsics.fx << ", fy=" << intrinsics.fy
            << ", cx=" << intrinsics.cx << ", cy=" << intrinsics.cy
            << "): all values must be finite, fx > 0 and fy != 0";
    throw InvalidInputError(message.str());
  }
}

std::string rigid_transform_error(const RigidTransform& transform) {
  for (double value : transform.m) {
    if (!std::isfinite(value)) {
      return "pose contains a non-finite value";
    }
  }

  const double bottom_row_deviation =
      std::fabs(transform.at(3, 0)) + std::fabs(transform.at(3, 1)) +
      std::fabs(transform.at(3, 2)) + std::fabs(transform.at(3, 3) - 1.0);
  if (bottom_row_deviation > kPoseBottomRowTolerance) {
    std::ostringstream message;
    message << "pose bottom row must be [0 0 0 1], got [" << transform.at(3, 0) << " "
            << transform.at(3, 1) << " " << transform.at(3, 2) << " " << transform.at(3, 3)
            << "] (row-major 4x4 expected)";
    return message.str();
  }

  double max_orthogonality_error = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double dot = 0.0;
      for (int k = 0; k < 3; ++k) {
        dot += transform.at(k, i) * transform.at(k, j);
      }
      const double expected = (i == j) ? 1.0 : 0.0;
      max_orthogonality_error = std::fmax(max_orthogonality_error, std::fabs(dot - expected));
    }
  }
  if (max_orthogonality_error > kRotationOrthogonalityTolerance) {
    std::ostringstream message;
    message << "pose rotation is not orthonormal: max |R^T R - I| = " << max_orthogonality_error;
    return message.str();
  }

  const double det =
      transform.at(0, 0) *
          (transform.at(1, 1) * transform.at(2, 2) - transform.at(1, 2) * transform.at(2, 1)) -
      transform.at(0, 1) *
          (transform.at(1, 0) * transform.at(2, 2) - transform.at(1, 2) * transform.at(2, 0)) +
      transform.at(0, 2) *
          (transform.at(1, 0) * transform.at(2, 1) - transform.at(1, 1) * transform.at(2, 0));
  if (std::fabs(det - 1.0) > kRotationDeterminantTolerance) {
    std::ostringstream message;
    message << "pose rotation determinant must be +1, got " << det;
    return message.str();
  }
  return {};
}

}  // namespace cudepthfusion
