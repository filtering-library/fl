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


#include <gtest/gtest.h>
#include "../typecast.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <iostream>

#include <fl/util/math/linear_algebra.hpp>
#include <fl/filter/filter_interface.hpp>

#include <fl/model/process/linear_process_model.hpp>
#include <fl/model/observation/linear_observation_model.hpp>
#include <fl/filter/gaussian/gaussian_filter.hpp>

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

    template <typename Filter>
    void setup_models(Filter& filter)
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

    State zero_state() { return State::Zero(StateDim); }
    Input zero_input() { return Input::Zero(InputDim); }
    Obsrv zero_obsrv() { return Obsrv::Zero(ObsrvDim); }

    State rand_state() { return State::Random(StateDim); }
    Input rand_input() { return Input::Random(InputDim); }
    Obsrv rand_obsrv() { return Obsrv::Random(ObsrvDim); }

    template <typename Filter, typename Belief>
    void predict(Filter& filter, Belief& belief)
    {
        EXPECT_TRUE(belief.mean().isZero());
        EXPECT_TRUE(belief.covariance().isIdentity());

        filter.predict(belief, zero_input(), belief);

        auto Q = filter.process_model().noise_matrix_squared();

        EXPECT_TRUE(belief.mean().isZero());
        EXPECT_TRUE(fl::are_similar(belief.covariance(), 2. * Q));
    }

    template <typename Filter, typename Belief>
    void predict_update(Filter& filter, Belief& belief)
    {
        setup_models(filter);

        EXPECT_TRUE(belief.covariance().ldlt().isPositive());

        for (int i = 0; i < 2000; ++i)
        {
            filter.predict(belief, zero_input(), belief);
            ASSERT_TRUE(belief.covariance().ldlt().isPositive());

            filter.update(belief, rand_obsrv(), belief);
            ASSERT_TRUE(belief.covariance().ldlt().isPositive());
        }
    }

    template <typename Filter, typename Belief>
    void predict_and_update(Filter& filter, Belief& belief_A, Belief& belief_B)
    {
        setup_models(filter);

        EXPECT_TRUE(belief_A.covariance().ldlt().isPositive());
        EXPECT_TRUE(belief_B.covariance().ldlt().isPositive());

        for (int i = 0; i < 2000; ++i)
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
};
