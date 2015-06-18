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

#include <Eigen/Dense>

#include <cmath>
#include <iostream>

#include <fl/model/process/linear_process_model.hpp>
#include <fl/model/observation/linear_observation_model.hpp>
#include <fl/filter/filter_interface.hpp>
#include <fl/filter/gaussian/gaussian_filter.hpp>
#include <fl/util/math/linear_algebra.hpp>


using namespace fl;

TEST(KalmanFilterTests, init_fixed_size_predict)
{
    constexpr int dim_state = 10;
    constexpr int dim_obsrv = 20;
    constexpr int dim_input = 1;

    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, dim_state, 1> State;
    typedef Eigen::Matrix<Scalar, dim_input, 1> Input;
    typedef Eigen::Matrix<Scalar, dim_obsrv, 1> Obsrv;

    typedef LinearStateTransitionModel<State, Input> LinearProcess;
    typedef LinearObservationModel<Obsrv, State> LinearObservation;

    // the KalmanFilter
    typedef GaussianFilter<LinearProcess, LinearObservation> Filter;
    auto filter = Filter(LinearProcess(), LinearObservation());
    Filter::Belief state_dist;

    EXPECT_TRUE(state_dist.mean().isZero());
    EXPECT_TRUE(state_dist.covariance().isIdentity());

    filter.predict(1.0, Input(1), state_dist, state_dist);

    auto Q = filter.process_model().noise_matrix_squared();

    EXPECT_TRUE(state_dist.mean().isZero());
    EXPECT_TRUE(fl::are_similar(state_dist.covariance(), 2. * Q));
}

TEST(KalmanFilterTests, init_dynamic_size_predict)
{
    constexpr int dim_state = 10;
    constexpr int dim_obsrv = 20;
    constexpr int dim_input = 1;

    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> State;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Input;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Obsrv;

    typedef LinearStateTransitionModel<State, Input> LinearProcess;
    typedef LinearObservationModel<Obsrv, State> LinearObservation;

    // the KalmanFilter
    typedef GaussianFilter<LinearProcess, LinearObservation> Filter;

    auto filter = Filter(LinearProcess(dim_state, dim_input),
                         LinearObservation(dim_obsrv, dim_state));

    auto state_dist  = Filter::Belief(dim_state);

    EXPECT_TRUE(state_dist.mean().isZero());
    EXPECT_TRUE(state_dist.covariance().isIdentity());

    filter.predict(1.0, Input(1), state_dist, state_dist);

    auto Q = filter.process_model().noise_matrix_squared();

    EXPECT_TRUE(state_dist.mean().isZero());
    EXPECT_TRUE(fl::are_similar(state_dist.covariance(), 2. * Q));
}

TEST(KalmanFilterTests, fixed_size_predict_update)
{
    constexpr int dim_state = 6;
    constexpr int dim_obsrv = 6;
    constexpr int dim_input = 6;

    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, dim_state, 1> State;
    typedef Eigen::Matrix<Scalar, dim_obsrv, 1> Input;
    typedef Eigen::Matrix<Scalar, dim_input, 1> Obsrv;

    typedef LinearStateTransitionModel<State, Input> LinearProcess;
    typedef LinearObservationModel<Obsrv, State> LinearObservation;

    // the KalmanFilter
    typedef GaussianFilter<LinearProcess, LinearObservation> Filter;

    auto filter = Filter(LinearProcess(dim_state, dim_input),
                         LinearObservation(dim_obsrv, dim_state));

    auto state_dist  = Filter::Belief(dim_state);

    auto A = filter.process_model().create_dynamics_matrix();
    auto Q = filter.process_model().create_noise_matrix();

    auto H = filter.obsrv_model().create_sensor_matrix();
    auto R = filter.obsrv_model().create_noise_matrix();

    A.setRandom();
    H.setRandom();
    Q.setRandom();
    R.setRandom();

    filter.process_model().dynamics_matrix(A);
    filter.process_model().noise_matrix(Q);

    filter.obsrv_model().sensor_matrix(H);
    filter.obsrv_model().noise_matrix(R);

    EXPECT_TRUE(state_dist.covariance().ldlt().isPositive());

    for (int i = 0; i < 2000; ++i)
    {
        filter.predict(1.0, Input(), state_dist, state_dist);
        EXPECT_TRUE(state_dist.covariance().ldlt().isPositive());

        Obsrv y = Obsrv::Random(dim_obsrv);
        filter.update(y, state_dist, state_dist);
        EXPECT_TRUE(state_dist.covariance().ldlt().isPositive());
    }
}

TEST(KalmanFilterTests, dynamic_size_predict_update)
{
    constexpr int dim_state = 10;
    constexpr int dim_obsrv = 10;
    constexpr int dim_input = 10;

    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> State;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Input;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Obsrv;

    typedef LinearStateTransitionModel<State, Input> LinearProcess;
    typedef LinearObservationModel<Obsrv, State> LinearObservation;

    // the KalmanFilter
    typedef GaussianFilter<LinearProcess, LinearObservation> Filter;

    auto filter = Filter(LinearProcess(dim_state, dim_input),
                         LinearObservation(dim_obsrv, dim_state));

    auto state_dist  = Filter::Belief(dim_state);

    auto A = filter.process_model().create_dynamics_matrix();
    auto Q = filter.process_model().create_noise_matrix();

    auto H = filter.obsrv_model().create_sensor_matrix();
    auto R = filter.obsrv_model().create_noise_matrix();

    A.setRandom();
    H.setRandom();
    Q.setRandom();
    R.setRandom();

    filter.process_model().dynamics_matrix(A);
    filter.process_model().noise_matrix(Q);

    filter.obsrv_model().sensor_matrix(H);
    filter.obsrv_model().noise_matrix(R);

    EXPECT_TRUE(state_dist.covariance().ldlt().isPositive());

    for (int i = 0; i < 2000; ++i)
    {
        filter.predict(1.0, Input(10), state_dist, state_dist);
        EXPECT_TRUE(state_dist.covariance().ldlt().isPositive());

        Obsrv y = Obsrv::Random(dim_obsrv);
        filter.update(y, state_dist, state_dist);
        EXPECT_TRUE(state_dist.covariance().ldlt().isPositive());
    }
}
