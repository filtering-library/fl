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
 * \file t_distribution_test.cpp
 * \date August 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>
#include "../typecast.hpp"

#include <cmath>
#include <iostream>

#include <fl/distribution/t_distribution.hpp>

template <typename TestType>
class TDistributionTests
    : public testing::Test
{
public:
    typedef typename TestType::Parameter Configuration;
    typedef fl::ScalarMatrix Variate;
    typedef fl::TDistribution<Variate> TDistribution;

    enum: signed int
    {
        DegreesOfFreedom = Configuration::DegreesOfFreedom
    };
};

TYPED_TEST_CASE_P(TDistributionTests);

TYPED_TEST_P(TDistributionTests, initial_degrees_of_freedom)
{
    typedef TestFixture This;
    typedef typename This::TDistribution TDistribution;

    auto t_distr = TDistribution(This::DegreesOfFreedom);

    EXPECT_GE(This::DegreesOfFreedom, 1);
    EXPECT_DOUBLE_EQ(t_distr.degrees_of_freedom(), This::DegreesOfFreedom);
}

TYPED_TEST_P(TDistributionTests, degrees_of_freedom)
{
    typedef TestFixture This;
    typedef typename This::TDistribution TDistribution;

    auto t_distr = TDistribution(1);

    EXPECT_GE(This::DegreesOfFreedom, 1);

    EXPECT_EQ(t_distr.degrees_of_freedom(), 1);
    t_distr.degrees_of_freedom(This::DegreesOfFreedom);
    EXPECT_DOUBLE_EQ(t_distr.degrees_of_freedom(), This::DegreesOfFreedom);
}

TYPED_TEST_P(TDistributionTests, probability)
{
    typedef TestFixture This;
    typedef typename This::TDistribution TDistribution;

    auto t_distr = TDistribution(This::DegreesOfFreedom);

    for (int i = 0; i < 10000; ++i)
    {
        fl::Real r = fl::Real(std::rand())/fl::Real(RAND_MAX) * 10.;

        EXPECT_LE(t_distr.probability(r), 0.4);
        EXPECT_GT(t_distr.probability(r), 0.0);
    }
}

//TYPED_TEST_P(TDistributionTests, map_standard_uniform)
//{
//    typedef TestFixture This;
//    typedef typename This::TDistribution TDistribution;
//    typedef typename This::Variate Variate;

//    auto t_distr = TDistribution(This::DegreesOfFreedom);

//    struct DofPairs
//    {
//        struct Pair
//        {
//            ::fl::Real prob;
//            ::fl::Real value;
//        };

//        fl::Real dof;
//        std::vector<Pair> pairs;
//    };

//    std::vector<DofPairs> lookup_table =
//    {
//        {1, {{0.75, 1.000}, {0.80, 1.376}}}
//    };

//    for (auto dof_pairs: lookup_table)
//    {
//        t_distr.degrees_of_freedom(dof_pairs.dof);

//        for (auto pair: dof_pairs.pairs)
//        {
//            Variate n(fl::uniform_to_normal(pair.prob));

//            ASSERT_TRUE(
//                fl::check_epsilon_bounds(
//                    t_distr.map_standard_normal(n) - pair.value, 1.e-2));
//        }
//    }
//}


REGISTER_TYPED_TEST_CASE_P(TDistributionTests,
                           initial_degrees_of_freedom,
                           degrees_of_freedom,
                           probability);

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

INSTANTIATE_TYPED_TEST_CASE_P(TDistributionTestCases,
                              TDistributionTests,
                              TestTypes);
