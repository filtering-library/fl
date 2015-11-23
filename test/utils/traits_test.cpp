/*
 * This is part of the fl library, a C++ Bayesian filtering library
 * (https://github.com/filtering-library)
 *
 * Copyright (c) 2015 Max Planck Society,
 * 				 Autonomous Motion Department,
 * 			     Institute for Intelligent Systems
 *
 * This Source Code Form is subject to the terms of the MIT License (MIT).
 * A copy of the license can be found in the LICENSE file distributed with this
 * source code.
 */

/**
 * \file traits_test.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <fl/util/math.hpp>
#include <fl/util/traits.hpp>

#include <cmath>

TEST(TraitsTests, is_dynamic)
{
    EXPECT_TRUE(fl::IsDynamic<Eigen::Dynamic>());
    EXPECT_FALSE(fl::IsDynamic<10>());
}

TEST(TraitsTests, is_fixed)
{
    EXPECT_FALSE(fl::IsFixed<Eigen::Dynamic>());
    EXPECT_TRUE(fl::IsFixed<10>());
}

TEST(TraitsTests, DimensionOf_dynamic)
{
    EXPECT_EQ(fl::DimensionOf<Eigen::MatrixXd>(), 0);
    EXPECT_EQ(fl::DimensionOf<Eigen::VectorXd>(), 0);

    typedef Eigen::Matrix<double, 10, -1> PartiallyDynamicMatrix;
    EXPECT_EQ(fl::DimensionOf<PartiallyDynamicMatrix>(), 0);
}

TEST(TraitsTests, DimensionOf_fixed)
{
    EXPECT_EQ(fl::DimensionOf<Eigen::Matrix3d>(), 3);
    EXPECT_EQ(fl::DimensionOf<Eigen::Vector4d>(), 4);

    typedef Eigen::Matrix<double, 10, 10> FixedMatrix;
    EXPECT_EQ(fl::DimensionOf<FixedMatrix>(), 10);

    typedef Eigen::Matrix<double, 10, 1> FixedVector;
    EXPECT_EQ(fl::DimensionOf<FixedVector>(), 10);
}
