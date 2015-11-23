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
 * \file chi_squared_distribution_test.cpp
 * \date August 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>
#include "../typecast.hpp"

#include <cmath>
#include <iostream>

#include <fl/distribution/chi_squared.hpp>

template <typename TestType>
class ChiSquaredTests
    : public testing::Test
{
public:
    typedef typename TestType::Parameter Configuration;

    enum: signed int
    {
        DegreesOfFreedom = Configuration::DegreesOfFreedom
    };
};

TYPED_TEST_CASE_P(ChiSquaredTests);

TYPED_TEST_P(ChiSquaredTests, initial_degrees_of_freedom)
{
    typedef TestFixture This;

    auto chi2 = fl::ChiSquared(This::DegreesOfFreedom);

    EXPECT_GE(This::DegreesOfFreedom, 1);
    EXPECT_DOUBLE_EQ(chi2.degrees_of_freedom(), This::DegreesOfFreedom);
}

TYPED_TEST_P(ChiSquaredTests, degrees_of_freedom)
{
    typedef TestFixture This;

    auto chi2 = fl::ChiSquared(1);

    EXPECT_GE(This::DegreesOfFreedom, 1);

    EXPECT_EQ(chi2.degrees_of_freedom(), 1);
    chi2.degrees_of_freedom(This::DegreesOfFreedom);
    EXPECT_DOUBLE_EQ(chi2.degrees_of_freedom(), This::DegreesOfFreedom);
}

TYPED_TEST_P(ChiSquaredTests, probability)
{
    typedef TestFixture This;

    auto chi2 = fl::ChiSquared(This::DegreesOfFreedom);

    for (int i = 0; i < 10000; ++i)
    {
        fl::Real r = fl::Real(std::rand())/fl::Real(RAND_MAX) * 100.;

        if (This::DegreesOfFreedom > 1 || r > 1.)
        {
            EXPECT_LE(chi2.probability(r), 0.5);
            EXPECT_GT(chi2.probability(r), 0.0);
        }
    }
}

TYPED_TEST_P(ChiSquaredTests, map_standard_uniform)
{
    typedef TestFixture This;

    auto chi2 = fl::ChiSquared(This::DegreesOfFreedom);

    struct Pair
    {
        fl::Real prob;
        fl::Real value;
    };

    std::vector<std::vector<Pair>> lookup_table =
    {
    /* dof    prob,value   prob,value   prob,value   prob,value   prob,value  */
    /* 1 */ {{0.05,0.004}, {0.1, 0.02}, {0.5, 0.46}, {0.8, 1.64}, {0.99, 6.64}},
    /* 2 */ {{0.05, 0.10}, {0.1, 0.21}, {0.5, 1.39}, {0.8, 3.22}, {0.99, 9.21}},
    /* 3 */ {{0.05, 0.35}, {0.1, 0.58}, {0.5, 2.37}, {0.8, 4.64}, {0.99,11.34}},
    /* 4 */ {{0.05, 0.71}, {0.1, 1.06}, {0.5, 3.36}, {0.8, 5.99}, {0.99,13.28}},
    /* 5 */ {{0.05, 1.14}, {0.1, 1.61}, {0.5, 4.35}, {0.8, 7.29}, {0.99,15.09}},
    /* 6 */ {{0.05, 1.63}, {0.1, 2.20}, {0.5, 5.35}, {0.8, 8.56}, {0.99,16.81}},
    /* 7 */ {{0.05, 2.17}, {0.1, 2.83}, {0.5, 6.35}, {0.8, 9.80}, {0.99,18.48}},
    /* 8 */ {{0.05, 2.73}, {0.1, 3.49}, {0.5, 7.34}, {0.8,11.03}, {0.99,20.09}},
    /* 9 */ {{0.05, 3.32}, {0.1, 4.17}, {0.5, 8.34}, {0.8,12.24}, {0.99,21.67}},
    /*10 */ {{0.05, 3.94}, {0.1, 4.87}, {0.5, 9.34}, {0.8,13.44}, {0.99,23.21}}
    };

    fl::Real dof = 0;
    for (auto dof_pairs: lookup_table)
    {
        chi2.degrees_of_freedom(++dof);

        for (auto pair: dof_pairs)
        {
            ASSERT_TRUE(
                fl::check_epsilon_bounds(
                    chi2.map_standard_uniform(pair.prob) - pair.value, 1.e-2));
        }
    }
}


REGISTER_TYPED_TEST_CASE_P(ChiSquaredTests,
                           initial_degrees_of_freedom,
                           degrees_of_freedom,
                           probability,
                           map_standard_uniform);

template <int DOF>
struct TestConfiguration
{
    enum: signed int { DegreesOfFreedom = DOF };
};

typedef ::testing::Types<
            fl::StaticTest<TestConfiguration<1>>,
            fl::StaticTest<TestConfiguration<2>>,
            fl::StaticTest<TestConfiguration<3>>,
            fl::StaticTest<TestConfiguration<10>>
        > TestTypes;

INSTANTIATE_TYPED_TEST_CASE_P(ChiSquaredTestCases,
                              ChiSquaredTests,
                              TestTypes);
