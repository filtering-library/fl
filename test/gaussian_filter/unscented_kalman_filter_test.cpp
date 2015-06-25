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
 * \file gaussian_filter_kf_test.cpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>
#include "../typecast.hpp"

#include <Eigen/Dense>

#include "gaussian_filter_test_suite.hpp"

#include <fl/filter/gaussian/unscented_transform.hpp>
#include <fl/filter/gaussian/gaussian_filter_ukf.hpp>

/**
 * Customize the GaussianFilterTest fixture by defining the Kalman filter
 */
template <typename TestType>
class UnscentedKalmanFilterTest
    : public GaussianFilterTest<TestType>
{
public:
    typedef GaussianFilterTest<TestType> Base;

    typedef typename Base::LinearStateTransition LinearStateTransition;
    typedef typename Base::LinearObservation LinearObservation;

    typedef fl::GaussianFilter<
                LinearStateTransition,
                LinearObservation,
                fl::UnscentedTransform
            > Filter;

    static Filter create_unscented_kalman_filter()
    {
        auto filter = Filter(
                        LinearStateTransition(Base::StateDim, Base::InputDim),
                        LinearObservation(Base::ObsrvDim, Base::StateDim),
                        fl::UnscentedTransform());

        return filter;
    }
};

typedef ::testing::Types<
            fl::StaticTest<>,
            fl::DynamicTest<>
        > TestTypes;

TYPED_TEST_CASE(UnscentedKalmanFilterTest, TestTypes);

TYPED_TEST(UnscentedKalmanFilterTest, init_predict)
{
    auto filter = TestFixture::create_unscented_kalman_filter();
    predict(filter);
}

TYPED_TEST(UnscentedKalmanFilterTest, predict_then_update)
{
    auto filter = TestFixture::create_unscented_kalman_filter();
    predict_update(filter);
}

TYPED_TEST(UnscentedKalmanFilterTest, predict_and_update)
{
    auto filter = TestFixture::create_unscented_kalman_filter();
    predict_and_update(filter);
}

TYPED_TEST(UnscentedKalmanFilterTest, predict_loop)
{
    auto filter = TestFixture::create_unscented_kalman_filter();
    predict_loop(filter);
}

TYPED_TEST(UnscentedKalmanFilterTest, predict_multiple_function_loop)
{
    auto filter = TestFixture::create_unscented_kalman_filter();
    predict_multiple_function_loop(filter);
}

TYPED_TEST(UnscentedKalmanFilterTest, predict_multiple)
{
    auto filter = TestFixture::create_unscented_kalman_filter();
    predict_multiple(filter);
}

TYPED_TEST(UnscentedKalmanFilterTest, predict_loop_vs_predict_multiple)
{
    auto filter = TestFixture::create_unscented_kalman_filter();
    predict_loop_vs_predict_multiple(filter);
}
