/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Max-Planck-Institute for Intelligent Systems,
 *                     University of Southern California
 *    Jan Issac (jan.issac@gmail.com)
 *    Manuel Wuthrich (manuel.wuthrich@gmail.com)
 *
 *
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

/**
 * @date 2015
 * @author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * Max-Planck-Institute for Intelligent Systems
 */

#include <gtest/gtest.h>

#include <fl/util/types.hpp>
#include <fl/model/process/binary_transition_density.hpp>


fl::FloatingPoint epsilon = 0.000000000001;
fl::FloatingPoint large_dt = 9999999999999.;



TEST(binary_transition_density, unit_delta_time)
{
    fl::FloatingPoint p_1to1 = 0.6;
    fl::FloatingPoint p_0to1 = 0.3;
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
    fl::FloatingPoint p_1to1 = 0.6;
    fl::FloatingPoint p_0to1 = 0.3;
    fl::BinaryTransitionDensity density(p_1to1, p_0to1);

    // the limit for dt -> infinity
    fl::FloatingPoint limit = p_0to1 / (1. - p_1to1 + p_0to1);

    EXPECT_TRUE(std::fabs(density.probability(1,0,large_dt) - limit) < epsilon);
    EXPECT_TRUE(std::fabs(density.probability(1,1,large_dt) - limit) < epsilon);
}


TEST(binary_transition_density, consistency)
{
    fl::FloatingPoint p_1to1 = 0.6;
    fl::FloatingPoint p_0to1 = 0.3;
    fl::BinaryTransitionDensity density(p_1to1, p_0to1);

    fl::FloatingPoint dt = 0.1;
    int N_steps = 10;
    fl::FloatingPoint initial_p_1 = 0.5;

    fl::FloatingPoint p_1 = initial_p_1;
    for(int i = 0; i < N_steps; i++)
    {
        // step-wise computation
        p_1 =     density.probability(1, 1, dt) * p_1
                + density.probability(1, 0, dt) * (1.-p_1);

        // direct computation
        fl::FloatingPoint p_1_ =
                density.probability(1, 1, (i+1) * dt) * initial_p_1
              + density.probability(1, 0, (i+1) * dt) * (1.-initial_p_1);


        EXPECT_TRUE(std::fabs(p_1 - p_1_) < epsilon);
    }
}


TEST(binary_transition_density, constant_system)
{
    fl::FloatingPoint p_1to1 = 1.;
    fl::FloatingPoint p_0to1 = 0.;
    fl::BinaryTransitionDensity density(p_1to1, p_0to1);

    EXPECT_TRUE(std::fabs(density.probability(1,1,large_dt) - 1.) < epsilon);
    EXPECT_TRUE(std::fabs(density.probability(0,0,large_dt) - 1.) < epsilon);

    EXPECT_TRUE(density.probability(0,1,large_dt) < epsilon);
    EXPECT_TRUE(density.probability(1,0,large_dt) < epsilon);
}











