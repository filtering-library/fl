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
 * \file gaussian_filter_test_suite.hpp
 * \date June 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__TEST__GAUSSIAN_FILTER__GAUSSIAN_FILTER_TEST_SUITE_HPP
#define FL__TEST__GAUSSIAN_FILTER__GAUSSIAN_FILTER_TEST_SUITE_HPP

#include <gtest/gtest.h>
#include "../typecast.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <iostream>

#include <fl/util/math/linear_algebra.hpp>
#include <fl/filter/filter_interface.hpp>

#include <fl/model/process/linear_process_model.hpp>
#include <fl/model/observation/linear_observation_model.hpp>

template <typename TestType>
class GaussianFilterTest
    : public ::testing::Test
{
protected:
    enum: signed int
    {
        StateDim = 10,
        InputDim = 5,
        ObsrvDim = 20,

        StateSize = fl::TestSize<StateDim, TestType>::Value,
        InputSize = fl::TestSize<InputDim, TestType>::Value,
        ObsrvSize = fl::TestSize<ObsrvDim, TestType>::Value
    };


    typedef Eigen::Matrix<fl::Real, StateSize, 1> State;
    typedef Eigen::Matrix<fl::Real, InputSize, 1> Input;
    typedef Eigen::Matrix<fl::Real, ObsrvSize, 1> Obsrv;

    typedef fl::LinearStateTransitionModel<State, Input> LinearStateTransition;
    typedef fl::LinearObservationModel<Obsrv, State> LinearObservation;

    GaussianFilterTest()
        : predict_steps_(10000),
          predict_update_steps_(2000)
    { }

    template <typename Filter>
    void setup_models_randomly(Filter& filter)
    {
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
    }

    template <typename Filter>
    void setup_identity_models(Filter& filter)
    {
        auto A = filter.process_model().create_dynamics_matrix();
        auto Q = filter.process_model().create_noise_matrix();

        auto H = filter.obsrv_model().create_sensor_matrix();
        auto R = filter.obsrv_model().create_noise_matrix();

        A.setIdentity();
        H.setIdentity();
        Q.setIdentity();
        R.setIdentity();

        filter.process_model().dynamics_matrix(A);
        filter.process_model().noise_matrix(Q);

        filter.obsrv_model().sensor_matrix(H);
        filter.obsrv_model().noise_matrix(R);
    }

    State zero_state() { return State::Zero(StateDim); }
    Input zero_input() { return Input::Zero(InputDim); }
    Obsrv zero_obsrv() { return Obsrv::Zero(ObsrvDim); }

    State rand_state() { return State::Random(StateDim); }
    Input rand_input() { return Input::Random(InputDim); }
    Obsrv rand_obsrv() { return Obsrv::Random(ObsrvDim); }

    template <typename Filter>
    void predict(Filter& filter)
    {
        auto belief = filter.create_belief();

        EXPECT_TRUE(belief.mean().isZero());
        EXPECT_TRUE(belief.covariance().isIdentity());

        filter.predict(belief, zero_input(), belief);

        auto Q = filter.process_model().noise_matrix_squared();

        EXPECT_TRUE(belief.mean().isZero());
        EXPECT_TRUE(fl::are_similar(belief.covariance(), 2. * Q));
    }

    template <typename Filter>
    void predict_update(Filter& filter)
    {
        setup_models_randomly(filter);

        auto belief = filter.create_belief();

        EXPECT_TRUE(belief.covariance().ldlt().isPositive());

        for (int i = 0; i < predict_update_steps_; ++i)
        {
            filter.predict(belief, zero_input(), belief);
            ASSERT_TRUE(belief.covariance().ldlt().isPositive());

            filter.update(belief, rand_obsrv(), belief);
            ASSERT_TRUE(belief.covariance().ldlt().isPositive());
        }
    }

    template <typename Filter>
    void predict_and_update(Filter& filter)
    {
        setup_models_randomly(filter);

        auto belief_A = filter.create_belief();
        auto belief_B = filter.create_belief();

        EXPECT_TRUE(belief_A.covariance().ldlt().isPositive());
        EXPECT_TRUE(belief_B.covariance().ldlt().isPositive());

        for (int i = 0; i < predict_update_steps_; ++i)
        {
            const auto y = rand_obsrv();
            const auto u = zero_input();

            filter.predict(belief_A, u, belief_A);
            filter.update(belief_A, y, belief_A);

            filter.predict_and_update(belief_B, u, y, belief_B);

            ASSERT_TRUE(
                fl::are_similar(belief_A.mean(), belief_B.mean()));
            ASSERT_TRUE(
                fl::are_similar(belief_A.covariance(), belief_B.covariance()));

            ASSERT_TRUE(belief_A.covariance().ldlt().isPositive());
            ASSERT_TRUE(belief_B.covariance().ldlt().isPositive());
        }
    }

    template <typename Filter>
    void predict_loop(Filter& filter)
    {
        setup_identity_models(filter);

        auto belief = filter.create_belief();

        EXPECT_TRUE(belief.covariance().ldlt().isPositive());
        for (int i = 0; i < predict_steps_; ++i)
        {
            filter.predict(belief, zero_input(), belief);
        }
        EXPECT_TRUE(belief.covariance().ldlt().isPositive());
    }

    template <typename Filter>
    void predict_multiple_function_loop(Filter& filter)
    {
        setup_identity_models(filter);

        auto belief = filter.create_belief();

        EXPECT_TRUE(belief.covariance().ldlt().isPositive());
        for (int i = 0; i < predict_steps_; ++i)
        {
            filter.predict(belief, zero_input(), 1, belief);
        }
        EXPECT_TRUE(belief.covariance().ldlt().isPositive());
    }

    template <typename Filter>
    void predict_multiple(Filter& filter)
    {
        setup_identity_models(filter);

        auto belief = filter.create_belief();

        EXPECT_TRUE(belief.covariance().ldlt().isPositive());
        filter.predict(belief, zero_input(), predict_steps_, belief);
        EXPECT_TRUE(belief.covariance().ldlt().isPositive());
    }

    template <typename Filter>
    void predict_loop_vs_predict_multiple(Filter& filter)
    {
        setup_identity_models(filter);

        auto belief_A = filter.create_belief();
        auto belief_B = filter.create_belief();

        EXPECT_TRUE(belief_A.covariance().ldlt().isPositive());
        for (int i = 0; i < predict_update_steps_; ++i)
        {
            filter.predict(belief_A, zero_input(), belief_A);
        }
        filter.predict(belief_B, zero_input(), predict_update_steps_, belief_B);

        EXPECT_TRUE(belief_A.covariance().ldlt().isPositive());
        EXPECT_TRUE(belief_B.covariance().ldlt().isPositive());

        EXPECT_TRUE(
            fl::are_similar(belief_A.mean(), belief_B.mean()));
        EXPECT_TRUE(
            fl::are_similar(belief_A.covariance(), belief_B.covariance()));
    }

protected:
    int predict_steps_;
    int predict_update_steps_;
};

#endif
