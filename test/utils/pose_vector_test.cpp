/*
 * This is part of the FL library, a C++ Bayesian filtering library
 * (https://github.com/filtering-library)
 *
 * Copyright (c) 2014 Jan Issac (jan.issac@gmail.com)
 * Copyright (c) 2014 Manuel Wuthrich (manuel.wuthrich@gmail.com)
 *
 * Max-Planck Institute for Intelligent Systems, AMD Lab
 * University of Southern California, CLMC Lab
 *
 * This Source Code Form is subject to the terms of the MIT License (MIT).
 * A copy of the license can be found in the LICENSE file distributed with this
 * source code.
 */

/**
 * @date 2015
 * @author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * Max-Planck-Institute for Intelligent Systems
 */

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <fl/util/math/pose_vector.hpp>

using namespace fl;

typedef EulerVector::RotationMatrix RotationMatrix;
typedef EulerVector::AngleAxis AngleAxis;
typedef EulerVector::Quaternion Quaternion;
typedef Eigen::Matrix<Real, 4, 4>              HomogeneousMatrix;
typedef Eigen::Matrix<Real, 6, 1> Vector6d;
typedef Eigen::Matrix<Real, 3, 1> Vector3d;
typedef Eigen::Matrix<Real, 4, 1> Vector4d;




Real epsilon = 0.000000001;


TEST(pose_vector, equality)
{
    Vector6d vector = Vector6d::Random();
    PoseVector pose_vector = vector;

    EXPECT_TRUE(pose_vector.isApprox(vector));
}


TEST(pose_vector, position)
{
    Vector3d vector = Vector3d::Random();
    PoseVector pose_vector;
    pose_vector.position() = vector;

    EXPECT_TRUE(pose_vector.position().isApprox(vector));
}


TEST(pose_vector, euler_vector)
{
    Vector3d vector = Vector3d::Random();
    PoseVector pose_vector;
    pose_vector.euler_vector() = vector;

    EXPECT_TRUE(pose_vector.euler_vector().isApprox(vector));
}


TEST(pose_vector, quaternion)
{
    EulerVector euler = EulerVector::Random();
    PoseVector pose_vector;

    pose_vector.euler_vector().quaternion(euler.quaternion());

    EXPECT_TRUE(pose_vector.euler_vector().isApprox(euler));
}


TEST(pose_vector, get_homogeneous)
{
    PoseVector pose_vector = PoseVector::Random();

    Vector3d va = Vector3d::Random();
    Vector4d vb; vb.topRows(3) = va; vb(3) = 1;

    va = pose_vector.euler_vector().rotation_matrix() * va
            + pose_vector.position();

    vb = pose_vector.homogeneous_matrix() * vb;

    EXPECT_TRUE(va.isApprox(vb.topRows(3)));
}


TEST(pose_vector, set_homogeneous)
{
    PoseVector pose_vector1 = PoseVector::Random();
    PoseVector pose_vector2;
    pose_vector2.homogeneous_matrix(pose_vector1.homogeneous_matrix());


    EXPECT_TRUE(pose_vector1.isApprox(pose_vector2));
}


TEST(pose_vector, product)
{
    PoseVector v1 = PoseVector::Random();
    PoseVector v2 = PoseVector::Random();

    PoseVector correct_result;
    correct_result.euler_vector().rotation_matrix(
                                    v2.euler_vector().rotation_matrix() *
                                    v1.euler_vector().rotation_matrix()   );
    correct_result.position() =
            v2.euler_vector().rotation_matrix() * v1.position()
            + v2.position();

    PoseVector operator_result = v2 * v1;

    EXPECT_TRUE(correct_result.isApprox(operator_result));
}


TEST(pose_vector, inverse)
{
    PoseVector v = PoseVector::Random();

    PoseVector result = v * v.inverse();
    EXPECT_TRUE(result.norm() < 0.00001);

    result = v.inverse() * v;
    EXPECT_TRUE(result.norm() < 0.00001);
}










