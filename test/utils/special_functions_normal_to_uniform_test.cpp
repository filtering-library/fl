/*
 * This is part of the FL library, a C++ Bayesian filtering library
 * (https://github.com/filtering-library)
 *
 * Copyright (c) N14 Jan Issac (jan.issac@gmail.com)
 * Copyright (c) N14 Manuel Wuthrich (manuel.wuthrich@gmail.com)
 *
 * Max-Planck Institute for Intelligent Systems, AMD Lab
 * University of Southern California, CLMC Lab
 *
 * This Source Code Form is subject to the terms of the MIT License (MIT).
 * A copy of the license can be found in the LICENSE file distributed with this
 * source code.
 */

/**
 * \file special_functions_normal_to_uniform_test.cpp
 * \date August 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>

#include <fl/util/math/special_functions.hpp>
#include <fl/distribution/standard_gaussian.hpp>

TEST(SpecialFunctions, normal_to_uniform)
{
    fl::StandardGaussian<fl::Real> normal;

    for(int i = 0; i < 100000; ++i)
    {
        fl::Real r = normal.sample();
        EXPECT_GT(fl::normal_to_uniform(r), 0.0);
        EXPECT_LT(fl::normal_to_uniform(r), 1.0);
    }
}

TEST(SpecialFunctions, mean)
{
    fl::StandardGaussian<fl::Real> normal;

    fl::Real mean = 0;
    int iterations = 1e6;

    for(int i = 0; i < iterations; ++i)
    {
        mean += fl::normal_to_uniform(normal.sample());
    }

    mean /= fl::Real(iterations);

    EXPECT_GT(mean, 0.4);
    EXPECT_LT(mean, 0.6);
}
