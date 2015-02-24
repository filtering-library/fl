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
 * \file meta_test.hpp
 * \date Januray 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <fl/util/math.hpp>
#include <fl/util/traits.hpp>
#include <fl/util/meta.hpp>

#include <cmath>

using namespace fl;

TEST(MetaTests, ExpandSizes)
{
    EXPECT_EQ((ExpandSizes<-1, 10>::Size), -1);
    EXPECT_EQ((ExpandSizes<10, -1>::Size), -1);
    EXPECT_EQ((ExpandSizes<-1, -1>::Size), -1);

    EXPECT_EQ((ExpandSizes<1, 1>::Size), 1);
    EXPECT_EQ((ExpandSizes<1, 10>::Size), 10);

    EXPECT_EQ((ExpandSizes<2, 10>::Size), 20);
    EXPECT_EQ((ExpandSizes<3, 5>::Size), 15);
}


TEST(MetaTests, JoinSizes_single_param)
{
    EXPECT_EQ((JoinSizes<0>::Size), 0);
    EXPECT_EQ((JoinSizes<-1>::Size), -1);
    EXPECT_EQ((JoinSizes<1>::Size), 1);
    EXPECT_EQ((JoinSizes<2>::Size), 2);
}

TEST(MetaTests, JoinSizes_two_param)
{
    EXPECT_EQ((JoinSizes<-1, -1>::Size), -1);
    EXPECT_EQ((JoinSizes<-1,  1>::Size), -1);
    EXPECT_EQ((JoinSizes< 1, -1>::Size), -1);
    EXPECT_EQ((JoinSizes< 1,  1>::Size),  2);

    // EXPECT_EQ((JoinSizes<-1,  0>::Size), -1); // should be rejected
    // EXPECT_EQ((JoinSizes< 1,  0>::Size),  2); // should be rejected
}

TEST(MetaTests, JoinSizes_multiple_param)
{
    EXPECT_EQ((JoinSizes<-1, -1, 1, 2, -1>::Size), -1);
    EXPECT_EQ((JoinSizes< 1, -1, 1, 2, -1>::Size), -1);
    EXPECT_EQ((JoinSizes< 1,  1, 1, 2, -1>::Size), -1);
    EXPECT_EQ((JoinSizes<-1,  1, 1, 2, -1>::Size), -1);
    EXPECT_EQ((JoinSizes< 1, -1, 1, 2, -1>::Size), -1);
    EXPECT_EQ((JoinSizes<1, 1, 1, 1, 1>::Size), 5);
    EXPECT_EQ((JoinSizes<1, 2, 3, 4, 5>::Size), 5*6/2);
}


TEST(MetaTests, CombineOptions_NoOptions)
{
    EXPECT_EQ(CombineOptions<>::Options::Value, NoOptions);
    EXPECT_EQ(CombineOptions<NoOptions>::Options::Value, NoOptions);

    EXPECT_EQ(
        CombineOptions<>::Options::Value, Options<NoOptions>::Value);
    EXPECT_EQ(
        CombineOptions<NoOptions>::Options::Value, Options<NoOptions>::Value);
}


TEST(MetaTests, CombineOptions_single_argument)
{
    EXPECT_EQ(
        CombineOptions<AdditiveProcessNoise>::Options::Value,
        AdditiveProcessNoise
    );

    EXPECT_EQ(
        CombineOptions<AdditiveProcessNoise>::Options::Value,
        Options<AdditiveProcessNoise>::Value
    );
}

TEST(MetaTests, CombineOptions_two_argument)
{
    EXPECT_EQ(
        (CombineOptions<AdditiveProcessNoise,
         AdditiveObsrvNoise>::Options::Value),
        AdditiveProcessNoise | AdditiveObsrvNoise
    );

    EXPECT_EQ(
        (CombineOptions<AdditiveProcessNoise,
         AdditiveObsrvNoise>::Options::Value),
        Options<AdditiveProcessNoise | AdditiveObsrvNoise>::Value
    );
}

TEST(MetaTests, CombineOptions_three_argument)
{
    EXPECT_EQ(
        (CombineOptions<
            AdditiveProcessNoise,
            AdditiveObsrvNoise,
            FactorizeParams>::Options::Value),
        AdditiveProcessNoise | AdditiveObsrvNoise | FactorizeParams
    );

    EXPECT_EQ(
        (CombineOptions<
             AdditiveProcessNoise,
             AdditiveObsrvNoise,
             FactorizeParams>::Options::Value),
        Options<
            AdditiveProcessNoise    |
            AdditiveObsrvNoise      |
            FactorizeParams>::Value
    );
}

