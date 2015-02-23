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

TEST(KalmanFilterTests, init_fixed_size_predict)
{
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, 10, 1> State;
    typedef Eigen::Matrix<Scalar, 1, 1> Input;
    typedef Eigen::Matrix<Scalar, 20, 1> Observation;

    // the KalmanFilter
    typedef fl::GaussianFilter<
                fl::LinearGaussianProcessModel<State, Input>,
                fl::LinearGaussianObservationModel<Observation, State>
            > Filter;

    typedef typename fl::Traits<Filter>::ProcessModel ProcessModel;
    typedef typename fl::Traits<Filter>::ObservationModel ObservationModel;

    ProcessModel::SecondMoment Q = ProcessModel::SecondMoment::Identity();
    ObservationModel::SecondMoment R = ObservationModel::SecondMoment::Identity();

    std::shared_ptr<ProcessModel> process_model =
            std::make_shared<ProcessModel>(Q);
    std::shared_ptr<ObservationModel> observation_model =
            std::make_shared<ObservationModel>(R);

    fl::FilterInterface<Filter>::Ptr filter =
            std::make_shared<Filter>(process_model, observation_model);

    Filter::StateDistribution state_dist;

    EXPECT_TRUE(state_dist.mean().isZero());
    EXPECT_TRUE(state_dist.covariance().isIdentity());
    filter->predict(1.0, Input(1), state_dist, state_dist);
    EXPECT_TRUE(state_dist.mean().isZero());
    EXPECT_TRUE(state_dist.covariance().isApprox(2. * Q));
}

TEST(KalmanFilterTests, init_dynamic_size_predict)
{
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> State;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Input;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Observation;

    const size_t dim_state = 10;
    const size_t dim_observation = 20;

    // the KalmanFilter
    typedef fl::GaussianFilter<
                fl::LinearGaussianProcessModel<State, Input>,
                fl::LinearGaussianObservationModel<Observation, State>
            > Filter;

    typedef typename fl::Traits<Filter>::ProcessModel ProcessModel;
    typedef typename fl::Traits<Filter>::ObservationModel ObservationModel;

    ProcessModel::SecondMoment Q =
            ProcessModel::SecondMoment::Identity(dim_state, dim_state);
    ObservationModel::SecondMoment R =
            ObservationModel::SecondMoment::Identity(
                dim_observation, dim_observation);

    std::shared_ptr<ProcessModel> process_model =
            std::make_shared<ProcessModel>(Q, dim_state);
    std::shared_ptr<ObservationModel> observation_model =
            std::make_shared<ObservationModel>(R, dim_observation, dim_state);

    fl::FilterInterface<Filter>::Ptr filter =
            std::make_shared<Filter>(process_model, observation_model);

    Filter::StateDistribution state_dist(dim_state);    

    EXPECT_TRUE(state_dist.mean().isZero());
    EXPECT_TRUE(state_dist.covariance().isIdentity());

    filter->predict(1.0, Input(1), state_dist, state_dist);

    EXPECT_TRUE(state_dist.mean().isZero());
    EXPECT_TRUE(state_dist.covariance().isApprox(2. * Q));
}

TEST(KalmanFilterTests, fixed_size_predict_update)
{
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, 6, 1> State;
    typedef Eigen::Matrix<Scalar, 6, 1> Input;
    typedef Eigen::Matrix<Scalar, 6, 1> Observation;

    // the KalmanFilter
    typedef fl::GaussianFilter<
                fl::LinearGaussianProcessModel<State, Input>,
                fl::LinearGaussianObservationModel<Observation, State>
            > Filter;

    typedef typename fl::Traits<Filter>::ProcessModel ProcessModel;
    typedef typename fl::Traits<Filter>::ObservationModel ObservationModel;

    ProcessModel::SecondMoment Q = ProcessModel::SecondMoment::Random() * 1.5;
    ObservationModel::SecondMoment R = ObservationModel::SecondMoment::Random();

    Q *= Q.transpose();
    R *= R.transpose();

    ProcessModel::DynamicsMatrix A = ProcessModel::DynamicsMatrix::Random();
    ObservationModel::SensorMatrix H = ObservationModel::SensorMatrix::Random();

    std::shared_ptr<ProcessModel> process_model =
            std::make_shared<ProcessModel>(Q);
    std::shared_ptr<ObservationModel> observation_model =
            std::make_shared<ObservationModel>(R);

    process_model->A(A);
    observation_model->H(H);

    fl::FilterInterface<Filter>::Ptr filter =
            std::make_shared<Filter>(process_model, observation_model);

    Filter::StateDistribution state_dist;
    EXPECT_TRUE(state_dist.covariance().ldlt().isPositive());

    for (size_t i = 0; i < 2000; ++i)
    {
        filter->predict(1.0, Input(), state_dist, state_dist);
        EXPECT_TRUE(state_dist.covariance().ldlt().isPositive());
        Observation y = Observation::Random();
        filter->update(y, state_dist, state_dist);
        EXPECT_TRUE(state_dist.covariance().ldlt().isPositive());
    }
}

TEST(KalmanFilterTests, dynamic_size_predict_update)
{
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> State;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Input;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Observation;

    const size_t dim_state = 10;
    const size_t dim_observation = 10;

    // the KalmanFilter
    typedef fl::GaussianFilter<
                fl::LinearGaussianProcessModel<State, Input>,
                fl::LinearGaussianObservationModel<Observation, State>
            > Filter;

    typedef typename fl::Traits<Filter>::ProcessModel ProcessModel;
    typedef typename fl::Traits<Filter>::ObservationModel ObservationModel;

    ProcessModel::SecondMoment Q =
            ProcessModel::SecondMoment::Random(dim_state, dim_state)*1.5;
    Q *= Q.transpose();

    ProcessModel::DynamicsMatrix A =
            ProcessModel::DynamicsMatrix::Random(dim_state, dim_state);

    ObservationModel::SecondMoment R =
            ObservationModel::SecondMoment::Random(dim_observation, dim_observation);
    R *= R.transpose();

    ObservationModel::SensorMatrix H =
            ObservationModel::SensorMatrix::Random(dim_observation, dim_state);

    std::shared_ptr<ProcessModel> process_model =
            std::make_shared<ProcessModel>(Q, dim_state);
    std::shared_ptr<ObservationModel> observation_model =
            std::make_shared<ObservationModel>(R, dim_observation, dim_state);

    process_model->A(A);
    observation_model->H(H);

    fl::FilterInterface<Filter>::Ptr filter =
            std::make_shared<Filter>(process_model, observation_model);

    Filter::StateDistribution state_dist(dim_state);
    EXPECT_TRUE(state_dist.covariance().ldlt().isPositive());

    for (size_t i = 0; i < 2000; ++i)
    {
        filter->predict(1.0, Input(1), state_dist, state_dist);
        EXPECT_TRUE(state_dist.covariance().ldlt().isPositive());
        Observation y = Observation::Random(dim_observation);
        filter->update(y, state_dist, state_dist);
        EXPECT_TRUE(state_dist.covariance().ldlt().isPositive());
    }
}
