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
 * \file gaussian_test.cpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>
#include "../typecast.hpp"

#include <cmath>
#include <iostream>

#include <fl/distribution/gaussian.hpp>

template <typename TestType>
class GaussianTests
    : public testing::Test
{
public:
    typedef typename TestType::Parameter Configuration;

    enum: signed int
    {
        Dim = Configuration::Dim,
        Size = fl::TestSize<Dim, TestType>::Value
    };

    typedef Eigen::Matrix<fl::Real, Size, 1> Vector;

    template <typename Gaussian>
    void test_gaussian_covariance(Gaussian& gaussian)
    {
        typedef typename Gaussian::SecondMoment Covariance;

        Covariance covariance = Eigen::MatrixXd::Identity(
                                    gaussian.dimension(),
                                    gaussian.dimension());
        Covariance square_root = covariance;
        Covariance precision = covariance;

        // first verify standard gaussian
        {
            SCOPED_TRACE("Unchanged");

            test_gaussian_attributes(
                        gaussian, covariance, precision, square_root);
        }

        // set covariance and verify representations
        {
            SCOPED_TRACE("Covariance setter");

            covariance.setRandom();
            covariance *= covariance.transpose();
            square_root = fl::matrix_sqrt(covariance);

            precision = covariance.inverse();
            gaussian.covariance(covariance);

            test_gaussian_attributes(
                        gaussian, covariance, precision, square_root);
        }

        // set square root and verify representations
        {
            SCOPED_TRACE("SquareRoot setter");

            square_root.setRandom();
            covariance = square_root * square_root.transpose();
            precision = covariance.inverse();
            gaussian.square_root(square_root);
            test_gaussian_attributes(
                        gaussian, covariance, precision, square_root);
        }

        // set covariance and verify representations
        {
            SCOPED_TRACE("Precision setter");

            precision.setRandom();
            precision *= precision.transpose();
            covariance = precision.inverse();
            square_root = fl::matrix_sqrt(covariance);
            gaussian.precision(precision);
            test_gaussian_attributes(
                        gaussian, covariance, precision, square_root);
        }
    }

    template <typename Gaussian, typename Covariance>
    void test_gaussian_attributes(Gaussian& gaussian,
                                  const Covariance& covariance,
                                  const Covariance& precision,
                                  const Covariance& square_root)
    {
        EXPECT_GT(gaussian.dimension(), 0);

        EXPECT_TRUE(fl::are_similar(gaussian.covariance(), covariance));
        EXPECT_TRUE(fl::are_similar(gaussian.precision(), precision));
        EXPECT_TRUE((fl::are_similar(
            gaussian.square_root() * gaussian.square_root().transpose(),
            square_root * square_root.transpose())));
        EXPECT_TRUE(gaussian.has_full_rank());
    }
};

TYPED_TEST_CASE_P(GaussianTests);

TYPED_TEST_P(GaussianTests, dimension)
{
    typedef TestFixture This;
    typedef fl::Gaussian<typename This::Vector> Gaussian;

    auto gaussian = Gaussian(This::Dim);

    EXPECT_EQ(gaussian.dimension(), This::Dim);
    EXPECT_EQ(gaussian.standard_variate_dimension(), This::Dim);
    EXPECT_EQ(gaussian.mean().size(), This::Dim);
    EXPECT_EQ(gaussian.covariance().rows(), This::Dim);
    EXPECT_EQ(gaussian.precision().rows(), This::Dim);
    EXPECT_EQ(gaussian.square_root().rows(), This::Dim);

    auto noise = Gaussian::StandardVariate::Random(
                        gaussian.standard_variate_dimension(), 1).eval();

    EXPECT_EQ(gaussian.map_standard_normal(noise).size(), This::Dim);
}


TYPED_TEST_P(GaussianTests, standard_covariance)
{
    typedef TestFixture This;
    typedef fl::Gaussian<typename This::Vector> Gaussian;
    auto gaussian = Gaussian(This::Dim);

    This::test_gaussian_covariance(gaussian);
}

TYPED_TEST_P(GaussianTests, gaussian_covariance_dimension_init)
{
    typedef TestFixture This;
    typedef fl::Gaussian<typename This::Vector> Gaussian;

    auto gaussian = Gaussian();

    gaussian.dimension(This::Dim);
    EXPECT_NO_THROW(This::test_gaussian_covariance(gaussian));
}


TYPED_TEST_P(GaussianTests, gaussian_covariance_constructor_init)
{
    typedef TestFixture This;
    typedef fl::Gaussian<typename This::Vector> Gaussian;

    auto gaussian = Gaussian(This::Dim);
    EXPECT_NO_THROW(This::test_gaussian_covariance(gaussian));
}

TYPED_TEST_P(GaussianTests, dynamic_uninitialized_gaussian)
{
    typedef TestFixture This;
    typedef fl::Gaussian<typename This::Vector> Gaussian;

    auto gaussian = Gaussian();

    if (This::Size != Eigen::Dynamic)
    {
        EXPECT_NO_THROW(gaussian.covariance());
        EXPECT_NO_THROW(gaussian.precision());
        EXPECT_NO_THROW(gaussian.square_root());
    }
    else
    {
        EXPECT_THROW(gaussian.covariance(), fl::GaussianUninitializedException);
        EXPECT_THROW(gaussian.precision(), fl::GaussianUninitializedException);
        EXPECT_THROW(gaussian.square_root(), fl::GaussianUninitializedException);

        gaussian.dimension(1);
        EXPECT_NO_THROW(gaussian.covariance());
        EXPECT_NO_THROW(gaussian.precision());
        EXPECT_NO_THROW(gaussian.square_root());

        gaussian.dimension(0);
        EXPECT_THROW(gaussian.covariance(), fl::GaussianUninitializedException);
        EXPECT_THROW(gaussian.precision(), fl::GaussianUninitializedException);
        EXPECT_THROW(gaussian.square_root(), fl::GaussianUninitializedException);
    }
}

REGISTER_TYPED_TEST_CASE_P(GaussianTests,
                           dimension,
                           standard_covariance,
                           dynamic_uninitialized_gaussian,
                           gaussian_covariance_dimension_init,
                           gaussian_covariance_constructor_init);

template <int Dimension>
struct TestConfiguration
{
    enum: signed int { Dim = Dimension };
};

typedef ::testing::Types<
            fl::StaticTest<TestConfiguration<2>>,
            fl::StaticTest<TestConfiguration<3>>,
            fl::StaticTest<TestConfiguration<10>>,
            fl::StaticTest<TestConfiguration<100>>,
            fl::DynamicTest<TestConfiguration<2>>,
            fl::DynamicTest<TestConfiguration<3>>,
            fl::DynamicTest<TestConfiguration<10>>,
            fl::DynamicTest<TestConfiguration<100>>
        > TestTypes;

INSTANTIATE_TYPED_TEST_CASE_P(GaussianTestCases,
                              GaussianTests,
                              TestTypes);
