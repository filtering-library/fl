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
#include <fl/util/math/multi_block_vector.hpp>
#include <fl/util/math/pose_vector.hpp>


using namespace fl;

typedef EulerVector::RotationMatrix RotationMatrix;
typedef EulerVector::AngleAxis AngleAxis;
typedef EulerVector::Quaternion Quaternion;
typedef Eigen::Matrix<Real, 4, 4> HomogeneousMatrix;
typedef Eigen::Matrix<Real, 6, 1> Vector6d;
typedef Eigen::Matrix<Real, 3, 1> Vector3d;
typedef Eigen::Matrix<Real, 4, 1> Vector4d;
typedef PoseVector::Affine Affine;




Real epsilon = 0.000000001;




TEST(multi_block_vector, equality)
{

    ComposedVector<PoseBlock<Eigen::VectorXd>, Eigen::VectorXd> vector;

    vector.recount(3);

    vector.setZero();




    PoseVector pose1 = PoseVector::Random();

    vector.component(1).euler_vector() = pose1.euler_vector();



    std::cout << "count " << vector.count() << std::endl;
    std::cout << "vector " << vector.transpose() << std::endl;




//    PoseBlock<Eigen::VectorXd>::Derived shiznit;

//    MultiBlockVector<PoseBlock<Eigen::VectorXd>> shizzle;

//    EXPECT_TRUE(multi_block_vector.isApprox(vector));
}


//TEST(multi_block_vector, equality)
//{
//    Vector6d vector = Vector6d::Random();
//    PoseVector multi_block_vector = vector;

//    EXPECT_TRUE(multi_block_vector.isApprox(vector));
//}











