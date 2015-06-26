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
#include <fl/filter/gaussian/monte_carlo_transform.hpp>
#include <fl/filter/gaussian/gaussian_filter_ukf.hpp>

template <int StateDimension, int InputDimension, int ObsrvDimension>
struct UnscentedKalmanFilterTestConfiguration
{
    enum: signed int
    {
        StateDim = StateDimension,
        InputDim = InputDimension,
        ObsrvDim = ObsrvDimension
    };

    template <typename StateTransitionModel, typename ObservationModel>
    struct FilterDefinition
    {
        typedef fl::GaussianFilter<
                    StateTransitionModel,
                    ObservationModel,
                    fl::UnscentedTransform
                > Type;
    };

    template <typename F, typename H>
    static typename FilterDefinition<F, H>::Type create_filter(F&& f, H&& h)
    {
        return typename FilterDefinition<F, H>::Type(f, h, fl::UnscentedTransform());
    }
};

typedef ::testing::Types<
            fl::StaticTest<UnscentedKalmanFilterTestConfiguration<3, 1, 2>>,
            fl::StaticTest<UnscentedKalmanFilterTestConfiguration<3, 3, 10>>,
            fl::StaticTest<UnscentedKalmanFilterTestConfiguration<10, 10, 20>>,

            fl::DynamicTest<UnscentedKalmanFilterTestConfiguration<3, 1, 2>>,
            fl::DynamicTest<UnscentedKalmanFilterTestConfiguration<3, 3, 10>>,
            fl::DynamicTest<UnscentedKalmanFilterTestConfiguration<10, 10, 20>>
        > TestTypes;

INSTANTIATE_TYPED_TEST_CASE_P(UnscentedKalmanFilterTest,
                              GaussianFilterTest,
                              TestTypes);
