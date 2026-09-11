#pragma once

#include <string>

#include "cudepthfusion/types.hpp"

namespace cudepthfusion {

inline constexpr double kPoseBottomRowTolerance = 1e-9;
inline constexpr double kRotationOrthogonalityTolerance = 1e-6;
inline constexpr double kRotationDeterminantTolerance = 1e-6;

// Throws InvalidInputError if fx is not positive, fy is zero, or any value is non-finite.
void validate_intrinsics(const Intrinsics& intrinsics);

// Returns an empty string when T is a finite rigid transform (bottom row [0 0 0 1],
// R^T R = I, det R = +1 within tolerance); otherwise a human-readable reason.
std::string rigid_transform_error(const RigidTransform& transform);

}  // namespace cudepthfusion
