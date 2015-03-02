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
 * \file options_test.hpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>

#include <fl/util/meta/options_argument.hpp>

using namespace fl;

TEST(MetaOptionsTests, CombineOptions_NoOptions)
{
    EXPECT_EQ(CombineOptions<>::Options::Value, NoOptions);
    EXPECT_EQ(CombineOptions<NoOptions>::Options::Value, NoOptions);

    EXPECT_EQ(
        CombineOptions<>::Options::Value, Options<NoOptions>::Value);
    EXPECT_EQ(
        CombineOptions<NoOptions>::Options::Value, Options<NoOptions>::Value);
}


TEST(MetaOptionsTests, CombineOptions_single_argument)
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

TEST(MetaOptionsTests, CombineOptions_two_argument)
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

TEST(MetaOptionsTests, CombineOptions_three_argument)
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
