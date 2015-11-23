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
 * \file gaussian_filter_linear_vs_unscented_kalman_filter_test.cpp
 * \date July 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <fl/util/meta.hpp>
#include <fl/filter/gaussian/gaussian_filter.hpp>

#include "../typecast.hpp"
#include "gaussian_filter_linear_vs_x_test_suite.hpp"

template <
    int StateDimension,
    int InputDimension,
    int ObsrvDimension,
    int ModelKey,
    template <typename> class StateTransitionModelNoiseType,
    template <typename> class ObsrvModelNoiseType
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
        typedef fl::GaussianFilter<
                    StateTransitionModel,
                    ObservationModel,
                    fl::UnscentedQuadrature,
                    fl::SigmaPointPredictPolicy<
                        fl::UnscentedQuadrature,
                        typename fl::UseAs<
                            StateTransitionModelNoiseType<StateTransitionModel>
                        >::Type>,
                    fl::SigmaPointUpdatePolicy<
                        fl::UnscentedQuadrature,
                        typename fl::UseAs<
                            ObsrvModelNoiseType<ObservationModel>
                        >::Type>
                > Type;
    };

    template <typename F, typename H>
    static typename FilterDefinition<F, H>::Type create_filter(F&& f, H&& h)
    {
        return typename FilterDefinition<F, H>::Type(
            f, h, fl::UnscentedQuadrature());
    }
};
