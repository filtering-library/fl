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

#include <fl/model/process/linear_state_transition_model.hpp>
#include <fl/model/observation/linear_gaussian_observation_model.hpp>
#include <fl/model/observation/linear_decorrelated_gaussian_observation_model.hpp>

template <typename TestType>
class GaussianFilterTest
    : public ::testing::Test
{
protected:
    typedef typename TestType::Parameter Configuration;

    enum: signed int
    {
        StateDim = Configuration::StateDim,
        InputDim = Configuration::InputDim,
        ObsrvDim = Configuration::ObsrvDim,

        StateSize = fl::TestSize<StateDim, TestType>::Value,
        InputSize = fl::TestSize<InputDim, TestType>::Value,
        ObsrvSize = fl::TestSize<ObsrvDim, TestType>::Value
    };

    enum ModelSetup
    {
        Random,
        Identity
    };

    typedef Eigen::Matrix<fl::Real, StateSize, 1> State;
    typedef Eigen::Matrix<fl::Real, InputSize, 1> Input;
    typedef Eigen::Matrix<fl::Real, ObsrvSize, 1> Obsrv;

    typedef fl::LinearStateTransitionModel<State, Input> LinearStateTransition;
    typedef fl::LinearDecorrelatedGaussianObservationModel<Obsrv, State> LinearObservation;
    typedef fl::LinearGaussianObservationModel<Obsrv, State> LinearObservation;


    typedef typename Configuration::template FilterDefinition<
                LinearStateTransition,
                LinearObservation
            > FilterDefinition;

    typedef typename FilterDefinition::Type Filter;

    GaussianFilterTest()
        : predict_steps_(30),
          predict_update_steps_(30)
    { }

    Filter create_filter() const
    {
        return Configuration::create_filter(
                LinearStateTransition(StateDim, InputDim),
                LinearObservation(ObsrvDim, StateDim));
    }

    void setup_models(Filter& filter, ModelSetup setup)
    {
        auto A = filter.process_model().create_dynamics_matrix();
        auto Q = filter.process_model().create_noise_matrix();

        auto H = filter.obsrv_model().create_sensor_matrix();
        auto R = filter.obsrv_model().create_noise_matrix();

        switch (setup)
        {
        case Random:
            A.setRandom();
            H.setRandom();
            Q.setRandom();
            R.setRandom();
            break;

        case Identity:
            A.setIdentity();
            H.setIdentity();
            Q.setIdentity();
            R.setIdentity();
            break;
        }

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

protected:
    int predict_steps_;
    int predict_update_steps_;
};

TYPED_TEST_CASE_P(GaussianFilterTest);

TYPED_TEST_P(GaussianFilterTest, init_predict)
{
    typedef TestFixture This;

    auto filter = This::create_filter();
    auto belief = filter.create_belief();

    std::cout << "filter.name():\n\n"
              << filter.name() << std::endl;
    std::cout << "\nfilter.description():\n\n"
              << filter.description() << std::endl;

    EXPECT_TRUE(belief.mean().isZero());
    EXPECT_TRUE(belief.covariance().isIdentity());

    filter.predict(belief, This::zero_input(), belief);

    auto Q = filter.process_model().noise_covariance();

    EXPECT_TRUE(belief.mean().isZero());
    EXPECT_TRUE(fl::are_similar(belief.covariance(), 2. * Q));
}

TYPED_TEST_P(GaussianFilterTest, predict_then_update)
{
    typedef TestFixture This;

    auto filter = This::create_filter();
    This::setup_models(filter, This::Random);

    auto belief = filter.create_belief();

    EXPECT_TRUE(belief.covariance().ldlt().isPositive());

    for (int i = 0; i < This::predict_update_steps_; ++i)
    {
        filter.predict(belief, This::zero_input(), belief);
        ASSERT_TRUE(belief.covariance().ldlt().isPositive());

        filter.update(belief, This::rand_obsrv(), belief);
        ASSERT_TRUE(belief.covariance().ldlt().isPositive());
    }
}

//TYPED_TEST_P(GaussianFilterTest, predict_and_update)
//{
//    typedef TestFixture This;

//    auto filter = This::create_filter();
//    This::setup_models(filter, This::Random);

//    auto belief_A = filter.create_belief();
//    auto belief_B = filter.create_belief();

//    EXPECT_TRUE(belief_A.covariance().ldlt().isPositive());
//    EXPECT_TRUE(belief_B.covariance().ldlt().isPositive());

//    for (int i = 0; i < This::predict_update_steps_; ++i)
//    {
//        const auto y = This::rand_obsrv();
//        const auto u = This::zero_input();

//        filter.predict(belief_A, u, belief_A);
//        filter.update(belief_A, y, belief_A);

//        filter.predict_and_update(belief_B, u, y, belief_B);

//        ASSERT_TRUE(
//            fl::are_similar(belief_A.mean(), belief_B.mean()));
//        ASSERT_TRUE(
//            fl::are_similar(belief_A.covariance(), belief_B.covariance()));

//        ASSERT_TRUE(belief_A.covariance().ldlt().isPositive());
//        ASSERT_TRUE(belief_B.covariance().ldlt().isPositive());
//    }
//}

TYPED_TEST_P(GaussianFilterTest, predict_loop)
{
    typedef TestFixture This;

    auto filter = This::create_filter();
    This::setup_models(filter, This::Identity);

    auto belief = filter.create_belief();

    EXPECT_TRUE(belief.covariance().ldlt().isPositive());

    for (int i = 0; i < This::predict_steps_; ++i)
    {
        filter.predict(belief, This::zero_input(), belief);
    }

    EXPECT_TRUE(belief.covariance().ldlt().isPositive());
}

//TYPED_TEST_P(GaussianFilterTest, predict_multiple_function_loop)
//{
//    typedef TestFixture This;

//    auto filter = This::create_filter();
//    This::setup_models(filter, This::Identity);

//    auto belief = filter.create_belief();

//    EXPECT_TRUE(belief.covariance().ldlt().isPositive());

//    for (int i = 0; i < This::predict_steps_; ++i)
//    {
//        filter.predict(belief, This::zero_input(), 1, belief);
//    }

//    EXPECT_TRUE(belief.covariance().ldlt().isPositive());
//}

//TYPED_TEST_P(GaussianFilterTest, predict_multiple)
//{
//    typedef TestFixture This;

//    auto filter = This::create_filter();
//    This::setup_models(filter, This::Identity);

//    auto belief = filter.create_belief();

//    EXPECT_TRUE(belief.covariance().ldlt().isPositive());

//    filter.predict(belief, This::zero_input(), This::predict_steps_, belief);

//    EXPECT_TRUE(belief.covariance().ldlt().isPositive());
//}

//TYPED_TEST_P(GaussianFilterTest, predict_loop_vs_predict_multiple)
//{
//    typedef TestFixture This;

//    auto filter = This::create_filter();
//    This::setup_models(filter, This::Identity);

//    auto belief_A = filter.create_belief();
//    auto belief_B = filter.create_belief();

//    EXPECT_TRUE(belief_A.covariance().ldlt().isPositive());

//    for (int i = 0; i < This::predict_update_steps_; ++i)
//    {
//        filter.predict(belief_A, This::zero_input(), belief_A);
//    }

//    filter.predict(belief_B,
//                   This::zero_input(),
//                   This::predict_update_steps_,
//                   belief_B);

//    EXPECT_TRUE(belief_A.covariance().ldlt().isPositive());
//    EXPECT_TRUE(belief_B.covariance().ldlt().isPositive());

//    EXPECT_TRUE(
//        fl::are_similar(belief_A.mean(), belief_B.mean()));
//    EXPECT_TRUE(
//        fl::are_similar(belief_A.covariance(), belief_B.covariance()));
//}

REGISTER_TYPED_TEST_CASE_P(GaussianFilterTest,
                           init_predict,
                           predict_then_update,
                           //predict_and_update,
                           predict_loop);
                           /*,
                           predict_multiple_function_loop,
                           predict_multiple,
                           predict_loop_vs_predict_multiple*/


#endif
