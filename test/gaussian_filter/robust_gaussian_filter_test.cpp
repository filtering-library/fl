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
 * \file robust_gaussian_filter_test.cpp
 * \date August 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "../typecast.hpp"
#include "gaussian_filter_test_suite.hpp"
#include <fl/util/meta.hpp>
#include <fl/filter/gaussian/gaussian_filter.hpp>
#include <fl/filter/gaussian/robust_gaussian_filter.hpp>

using namespace fl;

template <int StateDimension, int InputDimension, int ObsrvDimension>
struct RobutGaussianFilterTestConfiguration
{
    enum : signed int
    {
        StateDim = StateDimension,
        InputDim = InputDimension,
        ObsrvDim = ObsrvDimension
    };

    template <typename StateTransitionModel, typename ObservationModel>
    struct FilterDefinition
    {
        typedef RobustGaussianFilter<
                    StateTransitionModel,
                    ObservationModel,
                    UnscentedQuadrature
                > Type;
    };

    template <typename F, typename H>
    static typename FilterDefinition<F, H>::Type create_filter(F&& f, H&& h)
    {
        return typename FilterDefinition<F, H>::Type(f, h, UnscentedQuadrature());
    }
};

typedef ::testing::Types<
            StaticTest<RobutGaussianFilterTestConfiguration<3, 3, 3>>
        > TestTypes;

INSTANTIATE_TYPED_TEST_CASE_P(RobustGaussianFilterTest,
                              GaussianFilterTest,
                              TestTypes);
