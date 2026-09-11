#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>

#include "cudepthfusion/error.hpp"
#include "cudepthfusion/geometry.hpp"

namespace cudepthfusion {
namespace {

RigidTransform rotation_z_with_translation(double angle_rad, double tx, double ty, double tz) {
  RigidTransform t = RigidTransform::identity();
  const double c = std::cos(angle_rad);
  const double s = std::sin(angle_rad);
  // clang-format off
  t.m = {c,   -s,  0.0, tx,
         s,   c,   0.0, ty,
         0.0, 0.0, 1.0, tz,
         0.0, 0.0, 0.0, 1.0};
  // clang-format on
  return t;
}

TEST(Intrinsics, AcceptsPositiveAndNegativeFy) {
  EXPECT_NO_THROW(validate_intrinsics({481.2, 480.0, 319.5, 239.5}));
  EXPECT_NO_THROW(validate_intrinsics({481.2, -480.0, 319.5, 239.5}));
}

TEST(Intrinsics, RejectsDegenerateValues) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(validate_intrinsics({0.0, 480.0, 319.5, 239.5}), InvalidInputError);
  EXPECT_THROW(validate_intrinsics({-481.2, 480.0, 319.5, 239.5}), InvalidInputError);
  EXPECT_THROW(validate_intrinsics({481.2, 0.0, 319.5, 239.5}), InvalidInputError);
  EXPECT_THROW(validate_intrinsics({481.2, 480.0, nan, 239.5}), InvalidInputError);
}

TEST(RigidTransform, AcceptsIdentityAndProperRotation) {
  EXPECT_EQ(rigid_transform_error(RigidTransform::identity()), "");
  EXPECT_EQ(rigid_transform_error(rotation_z_with_translation(0.5, 1.0, -2.0, 3.0)), "");
}

TEST(RigidTransform, RejectsNonFinite) {
  RigidTransform t = RigidTransform::identity();
  t.m[3] = std::numeric_limits<double>::infinity();
  EXPECT_NE(rigid_transform_error(t).find("non-finite"), std::string::npos);
}

TEST(RigidTransform, RejectsColumnMajorTransform) {
  // Transposing a transform with translation moves t into the bottom row.
  const RigidTransform t = rotation_z_with_translation(0.1, 1.0, 2.0, 3.0);
  RigidTransform transposed;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      transposed.m[static_cast<std::size_t>(r * 4 + c)] = t.at(c, r);
    }
  }
  EXPECT_NE(rigid_transform_error(transposed).find("bottom row"), std::string::npos);
}

TEST(RigidTransform, RejectsScaledRotation) {
  RigidTransform t = RigidTransform::identity();
  t.m[0] = 1.01;
  EXPECT_NE(rigid_transform_error(t).find("orthonormal"), std::string::npos);
}

TEST(RigidTransform, RejectsReflection) {
  RigidTransform t = RigidTransform::identity();
  t.m[10] = -1.0;
  EXPECT_NE(rigid_transform_error(t).find("determinant"), std::string::npos);
}

}  // namespace
}  // namespace cudepthfusion
