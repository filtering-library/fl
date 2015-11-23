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
 * @date 2015
 * @author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * Max-Planck-Institute for Intelligent Systems
 */

#include <gtest/gtest.h>

#include <fl/util/types.hpp>
#include <fl/model/process/binary_transition_density.hpp>

fl::Real epsilon = 0.000000000001;
fl::Real large_dt = 9999999999999.;

TEST(binary_transition_density, unit_delta_time)
{
    fl::Real p_1to1 = 0.6;
    fl::Real p_0to1 = 0.3;
    fl::BinaryTransitionDensity density(p_1to1, p_0to1);

    EXPECT_TRUE(std::fabs(density.probability(1,1,1.) - p_1to1) < epsilon);
    EXPECT_TRUE(std::fabs(density.probability(1,0,1.) - p_0to1) < epsilon);
}

TEST(binary_transition_density, zero_delta_time)
{
    fl::BinaryTransitionDensity density(0.6, 0.3);

    EXPECT_TRUE(std::fabs(density.probability(1,1,0.) - 1.) < epsilon);
    EXPECT_TRUE(std::fabs(density.probability(0,0,0.) - 1.) < epsilon);

    EXPECT_TRUE(density.probability(1,0,0.) < epsilon);
    EXPECT_TRUE(density.probability(0,1,0.) < epsilon);
}

TEST(binary_transition_density, inifinite_delta_time)
{
    fl::Real p_1to1 = 0.6;
    fl::Real p_0to1 = 0.3;
    fl::BinaryTransitionDensity density(p_1to1, p_0to1);

    // the limit for dt -> infinity
    fl::Real limit = p_0to1 / (1. - p_1to1 + p_0to1);

    EXPECT_TRUE(std::fabs(density.probability(1,0,large_dt) - limit) < epsilon);
    EXPECT_TRUE(std::fabs(density.probability(1,1,large_dt) - limit) < epsilon);
}

TEST(binary_transition_density, consistency)
{
    fl::Real p_1to1 = 0.6;
    fl::Real p_0to1 = 0.3;
    fl::BinaryTransitionDensity density(p_1to1, p_0to1);

    fl::Real dt = 0.1;
    int N_steps = 10;
    fl::Real initial_p_1 = 0.5;

    fl::Real p_1 = initial_p_1;
    for(int i = 0; i < N_steps; i++)
    {
        // step-wise computation
        p_1 =     density.probability(1, 1, dt) * p_1
                + density.probability(1, 0, dt) * (1.-p_1);

        // direct computation
        fl::Real p_1_ =
                density.probability(1, 1, (i+1) * dt) * initial_p_1
              + density.probability(1, 0, (i+1) * dt) * (1.-initial_p_1);


        EXPECT_TRUE(std::fabs(p_1 - p_1_) < epsilon);
    }
}

TEST(binary_transition_density, constant_system)
{
    fl::Real p_1to1 = 1.;
    fl::Real p_0to1 = 0.;
    fl::BinaryTransitionDensity density(p_1to1, p_0to1);

    EXPECT_TRUE(std::fabs(density.probability(1,1,large_dt) - 1.) < epsilon);
    EXPECT_TRUE(std::fabs(density.probability(0,0,large_dt) - 1.) < epsilon);

    EXPECT_TRUE(density.probability(0,1,large_dt) < epsilon);
    EXPECT_TRUE(density.probability(1,0,large_dt) < epsilon);
}











