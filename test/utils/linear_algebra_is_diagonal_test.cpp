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
 * \file linear_algebra_is_diagonal_test.cpp
 * \date July 15
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>

#include <fl/util/types.hpp>
#include <fl/util/math/linear_algebra.hpp>

static constexpr int Dim = 10;

template <int Size>
void check_is_diagonal()
{
    Eigen::Matrix<fl::Real, Size, Size> m;

    m.resize(Dim, Dim);
    m.setRandom();

    EXPECT_FALSE(fl::is_diagonal(m));

    m = m.diagonal().asDiagonal();
    EXPECT_TRUE(fl::is_diagonal(m));

    m.setIdentity();
    m *= 3.;
    EXPECT_TRUE(fl::is_diagonal(m));

    m(0, Dim - 1) = 2;
    EXPECT_FALSE(fl::is_diagonal(m));
}

TEST(LinearAlgebra, is_diagonal_fixed_size)
{
    check_is_diagonal<Dim>();
}

TEST(LinearAlgebra, is_diagonal_dynamic_size)
{
    check_is_diagonal<Eigen::Dynamic>();
}
