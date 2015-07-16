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
#include <fl/util/math/rigid_body_state.hpp>

using namespace fl;

typedef EulerVector::RotationMatrix RotationMatrix;
typedef EulerVector::AngleAxis AngleAxis;
typedef EulerVector::Quaternion Quaternion;
typedef Eigen::Matrix<Real, 3, 1> Vector;

Real epsilon = 0.000000001;


TEST(rigid_body_state, euler_vector)
{
    RigidBodyState state = RigidBodyState::Zero();
    EulerVector euler_vector = EulerVector::Random();

    state.euler_vector().quaternion(euler_vector.quaternion());
    EXPECT_TRUE(state.euler_vector().isApprox(euler_vector));
}



TEST(rigid_body_state, angle_axis)
{
    Vector axis = Vector::Random();
    axis.normalize();
    Real angle = 1.155435;

    RigidBodyState state = RigidBodyState::Zero();
    state.euler_vector().angle_axis(angle, axis);

    EXPECT_TRUE(std::fabs(state.euler_vector().angle() - angle) < epsilon);
    EXPECT_TRUE(axis.isApprox(state.euler_vector().axis()));
}


TEST(rigid_body_state, consistency)
{
    Vector axis = Vector::Random();
    axis.normalize();
    Real angle = 1.155435;

    RigidBodyState state = RigidBodyState::Zero();
    state.euler_vector().angle_axis(angle, axis);

    EXPECT_TRUE(std::fabs(state.euler_vector().angle() - angle) < epsilon);
    EXPECT_TRUE(axis.isApprox(state.euler_vector().axis()));
}



