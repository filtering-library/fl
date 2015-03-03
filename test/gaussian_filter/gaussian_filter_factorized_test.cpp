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
 * \file gaussian_filter_factroized_test.cpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <fl/util/meta.hpp>
#include <fl/model/process/linear_process_model.hpp>
#include <fl/model/process/joint_process_model.hpp>
#include <fl/model/observation/linear_observation_model.hpp>

#include <fl/filter/filter_interface.hpp>
#include <fl/filter/gaussian/unscented_transform.hpp>
#include <fl/filter/gaussian/gaussian_filter_factorized.hpp>

TEST(GaussianFilterFactorizedTests, init)
{
    using namespace fl;

    typedef Eigen::Matrix<double, 3, 1> State;
    typedef Eigen::Matrix<double, 1, 1> Pixel;
    typedef Eigen::Matrix<double, 1, 1> Param;

    typedef LinearGaussianProcessModel<State> ProcessModel;
    typedef LinearGaussianProcessModel<Param> ParamModel;

    typedef NotAdaptive<LinearGaussianObservationModel<Pixel, State>> PixelModel;

    constexpr int pixels = 10;

    typedef GaussianFilter<
                JointProcessModel<
                    ProcessModel,
                    JointProcessModel<MultipleOf<ParamModel, pixels>>>,
                JointObservationModel<MultipleOf<PixelModel, pixels>>,
                UnscentedTransform,
                Options<FactorizeParams>
            > ExplicitFilter;

    typedef GaussianFilter<
                ProcessModel,
                Join<MultipleOf<Adaptive<PixelModel, ParamModel>, pixels>>,
                UnscentedTransform,
                Options<FactorizeParams>
            > AutoFilter;

    EXPECT_TRUE((std::is_base_of<ExplicitFilter, AutoFilter>::value));

    ExplicitFilter x = ExplicitFilter(
                           ProcessModel(),
                           ParamModel(),
                           PixelModel(),
                           UnscentedTransform(),
                           pixels);

    ExplicitFilter y = AutoFilter(
                           ProcessModel(),
                           ParamModel(),
                           PixelModel(),
                           UnscentedTransform(),
                           pixels);

}
