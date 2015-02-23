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
#include <fl/filter/gaussian/gaussian_filter.hpp>
//#include <fl/filter/gaussian/gaussian_filter_factorized.hpp>

TEST(GaussianFilterFactorizedTests, init)
{
//    using namespace fl;

//    typedef void ProcessModel;
//    typedef void PixelObsrvModel<typename ProcessModel::State>;
//    typedef void PixelParamModel<typename PixelObsrvModel::Param>;

//    typedef GaussianFilter<
//                ProcessModel,
//                Join<MultipleOf<AdaptiveModel<PixelObsrvModel, PixelParamModel>, 10>>
//            > Filter;



//    typedef GaussianFilter<
//                ProcessModel,
//                AdaptiveModel<PixelObsrvModel, PixelParamModel>
//            > Filter;

//    constexpr int state_dim = 5;
//    constexpr int obsrv_dim = 1;
//    constexpr int param_dim = 1;
//    constexpr int count = 10;

//    typedef Eigen::Matrix<double, state_dim, 1> State;
//    typedef Eigen::Matrix<double, obsrv_dim, 1> SingleObsrv;
//    typedef Eigen::Matrix<double, param_dim, 1> SingleParam;

//    typedef Eigen::Matrix<
//                double,
//                fl::JoinSizes<state_dim, param_dim>::Size,
//                1
//            > JointState;

//    typedef fl::LinearGaussianProcessModel<State> Process;
//    typedef fl::JointProcessModel<State, >


//    typedef fl::LinearGaussianProcessModel<SingleParam> SingleParamProcess;



//    typedef fl::LinearGaussianObservationModel<SingleObsrv> SingleObsrvModel;

//    typedef fl::GaussianFilter<>;


}
