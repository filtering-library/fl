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
 * \file gaussian_filter_linear_vs_unscented_kalman_filter_test.cpp
 * \date July 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>
#include "../typecast.hpp"

#include <Eigen/Dense>

#include "gaussian_filter_linear_vs_x_test_suite.hpp"
#include <fl/util/meta.hpp>
#include <fl/filter/gaussian/gaussian_filter.hpp>

using namespace fl;

template <
    int StateDimension,
    int InputDimension,
    int ObsrvDimension,
    int ModelKey
>
struct UkfTestConfig
{
    enum : signed int
    {
        StateDim = StateDimension,
        InputDim = InputDimension,
        ObsrvDim = ObsrvDimension
    };

    enum : signed int { SelectedModel = ModelKey };

    template <typename StateTransitionModel, typename ObservationModel>
    struct FilterDefinition
    {
        typedef GaussianFilter<
                    StateTransitionModel,
                    ObservationModel,
                    SigmaPointQuadrature<UnscentedTransform>/*,
                    SigmaPointPredictPolicy<
                        SigmaPointQuadrature<UnscentedTransform>,
                        typename UseAs<NonAdditive<StateTransitionModel>>::Type>,
                    SigmaPointUpdatePolicy<
                        SigmaPointQuadrature<UnscentedTransform>,
                        typename UseAs<NonAdditive<ObservationModel>>::Type>*/
                > Type;
    };

    template <typename F, typename H>
    static typename FilterDefinition<F, H>::Type create_filter(F&& f, H&& h)
    {
        return typename FilterDefinition<F, H>::Type(
            f, h, UnscentedQuadrature());
    }
};

typedef ::testing::Types<
            StaticTest<UkfTestConfig<4, 4, 4, GaussianModel>>,
            StaticTest<UkfTestConfig<3, 3, 10, GaussianModel>>,
            StaticTest<UkfTestConfig<10, 4, 10, GaussianModel>>,
            StaticTest<UkfTestConfig<20, 4, 2, GaussianModel>>,

            DynamicTest<UkfTestConfig<3, 1, 2, GaussianModel>>,
            DynamicTest<UkfTestConfig<3, 3, 3, GaussianModel>>,
            DynamicTest<UkfTestConfig<10, 4, 4, GaussianModel>>,
            DynamicTest<UkfTestConfig<20, 4, 2, GaussianModel>>,

            StaticTest<UkfTestConfig<3, 1, 2, DecorrelatedGaussianModel>>,
            StaticTest<UkfTestConfig<3, 3, 10, DecorrelatedGaussianModel>>,
            StaticTest<UkfTestConfig<20, 4, 2, DecorrelatedGaussianModel>>,
            StaticTest<UkfTestConfig<10, 3, 50, DecorrelatedGaussianModel>>,
            StaticTest<UkfTestConfig<10, 10, 500, DecorrelatedGaussianModel>>,

            DynamicTest<UkfTestConfig<3, 1, 2, DecorrelatedGaussianModel>>,
            DynamicTest<UkfTestConfig<3, 3, 10, DecorrelatedGaussianModel>>,
            DynamicTest<UkfTestConfig<10, 3, 50, DecorrelatedGaussianModel>>,
            DynamicTest<UkfTestConfig<20, 4, 2, DecorrelatedGaussianModel>>,
            DynamicTest<UkfTestConfig<10, 10, 500, DecorrelatedGaussianModel>>
        > TestTypes;

INSTANTIATE_TYPED_TEST_CASE_P(GaussianFilterLinearVUkfTest,
                              GaussianFilterLinearVsXTest,
                              TestTypes);
