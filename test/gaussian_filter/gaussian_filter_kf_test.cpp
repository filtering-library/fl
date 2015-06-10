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
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, 10, 1> State;
    typedef Eigen::Matrix<Scalar, 1 , 1> Input;
    typedef Eigen::Matrix<Scalar, 20, 1> Obsrv;

    // the KalmanFilter
    typedef GaussianFilter<
                LinearGaussianProcessModel<State, Input>,
                LinearGaussianObservationModel<Obsrv, State>
            > Algo;

    typedef FilterInterface<Algo> Filter;

    typedef typename Traits<Algo>::ProcessModel ProcessModel;
    typedef typename Traits<Algo>::ObservationModel ObservationModel;

    Traits<ProcessModel>::SecondMoment Q =
        Traits<ProcessModel>::SecondMoment::Identity();
    Traits<ObservationModel>::SecondMoment R =
        Traits<ObservationModel>::SecondMoment::Identity();

    Filter&& filter = Algo(ProcessModel(Q), ObservationModel(R));

    Filter::StateDistribution state_dist;

    EXPECT_TRUE(state_dist.mean().isZero());
    EXPECT_TRUE(state_dist.covariance().isIdentity());\

    filter.predict(1.0, Input(1), state_dist, state_dist);

    EXPECT_TRUE(state_dist.mean().isZero());
    EXPECT_TRUE(fl::are_similar(state_dist.covariance(), 2. * Q));
}

TEST(KalmanFilterTests, init_dynamic_size_predict)
{
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> State;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Input;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Obsrv;

    const size_t dim_state = 10;
    const size_t dim_obsrv = 20;

    // the KalmanFilter
    typedef GaussianFilter<
                LinearGaussianProcessModel<State, Input>,
                LinearGaussianObservationModel<Obsrv, State>
            > Algo;

    typedef FilterInterface<Algo> Filter;

    typedef typename Traits<Algo>::ProcessModel ProcessModel;
    typedef typename Traits<Algo>::ObservationModel ObservationModel;

    Traits<ProcessModel>::SecondMoment Q =
        Traits<ProcessModel>::SecondMoment::Identity(dim_state, dim_state);

    Traits<ObservationModel>::SecondMoment R =
        Traits<ObservationModel>::SecondMoment::Identity(dim_obsrv, dim_obsrv);

    Filter&& filter = Algo(
        ProcessModel(Q, dim_state),
        ObservationModel(R, dim_obsrv, dim_state));

    Filter::StateDistribution state_dist  = Filter::StateDistribution(dim_state);

    EXPECT_TRUE(state_dist.mean().isZero());
    EXPECT_TRUE(state_dist.covariance().isIdentity());

    filter.predict(1.0, Input(1), state_dist, state_dist);

    EXPECT_TRUE(state_dist.mean().isZero());
    EXPECT_TRUE(fl::are_similar(state_dist.covariance(), 2. * Q));
}

TEST(KalmanFilterTests, fixed_size_predict_update)
{
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, 6, 1> State;
    typedef Eigen::Matrix<Scalar, 6, 1> Input;
    typedef Eigen::Matrix<Scalar, 6, 1> Obsrv;

    // the KalmanFilter
    typedef GaussianFilter<
                LinearGaussianProcessModel<State, Input>,
                LinearGaussianObservationModel<Obsrv, State>
            > Algo;

    typedef FilterInterface<Algo> Filter;

    typedef typename Traits<Algo>::ProcessModel ProcessModel;
    typedef typename Traits<Algo>::ObservationModel ObservationModel;

    Traits<ProcessModel>::SecondMoment Q =
        Traits<ProcessModel>::SecondMoment::Random() * 1.5;
    Traits<ObservationModel>::SecondMoment R =
        Traits<ObservationModel>::SecondMoment::Random();

    Q *= Q.transpose();
    R *= R.transpose();

    Traits<ProcessModel>::DynamicsMatrix A =
        Traits<ProcessModel>::DynamicsMatrix::Random();

    Traits<ObservationModel>::SensorMatrix H =
        Traits<ObservationModel>::SensorMatrix::Random();

    ProcessModel process_model = ProcessModel(Q);
    ObservationModel obsrv_model = ObservationModel(R);

    process_model.A(A);
    obsrv_model.H(H);

    Filter&& filter = Algo(process_model, obsrv_model);

    Filter::StateDistribution state_dist;

    EXPECT_TRUE(state_dist.covariance().ldlt().isPositive());

    for (size_t i = 0; i < 2000; ++i)
    {
        filter.predict(1.0, Input(), state_dist, state_dist);
        EXPECT_TRUE(state_dist.covariance().ldlt().isPositive());

        Obsrv y = Obsrv::Random();
        filter.update(y, state_dist, state_dist);
        EXPECT_TRUE(state_dist.covariance().ldlt().isPositive());
    }
}

TEST(KalmanFilterTests, dynamic_size_predict_update)
{
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> State;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Input;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Obsrv;

    const size_t dim_state = 10;
    const size_t dim_obsrv = 10;

    // the KalmanFilter
    typedef GaussianFilter<
                LinearGaussianProcessModel<State, Input>,
                LinearGaussianObservationModel<Obsrv, State>
            > Algo;

    typedef FilterInterface<Algo> Filter;

    typedef typename Traits<Algo>::ProcessModel ProcessModel;
    typedef typename Traits<Algo>::ObservationModel ObservationModel;

    Traits<ProcessModel>::SecondMoment Q =
        Traits<ProcessModel>::SecondMoment::Random(dim_state, dim_state) * 1.5;
    Q *= Q.transpose();

    Traits<ProcessModel>::DynamicsMatrix A =
        Traits<ProcessModel>::DynamicsMatrix::Random(dim_state, dim_state);

    Traits<ObservationModel>::SecondMoment R =
        Traits<ObservationModel>::SecondMoment::Random(dim_obsrv, dim_obsrv);
    R *= R.transpose();

    Traits<ObservationModel>::SensorMatrix H =
        Traits<ObservationModel>::SensorMatrix::Random(dim_obsrv, dim_state);

    ProcessModel process_model =
        ProcessModel(Q, dim_state);

    ObservationModel obsrv_model =
        ObservationModel(R, dim_obsrv, dim_state);

    process_model.A(A);
    obsrv_model.H(H);

    Filter&& filter = Algo(process_model, obsrv_model);

    Filter::StateDistribution state_dist(dim_state);

    EXPECT_TRUE(state_dist.covariance().ldlt().isPositive());

    for (size_t i = 0; i < 2000; ++i)
    {
        filter.predict(1.0, Input(1), state_dist, state_dist);
        EXPECT_TRUE(state_dist.covariance().ldlt().isPositive());

        Obsrv y = Obsrv::Random(dim_obsrv);
        filter.update(y, state_dist, state_dist);
        EXPECT_TRUE(state_dist.covariance().ldlt().isPositive());
    }
}
