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
 * \file gaussian_filter_linear_vs_x_test_suite.hpp
 * \date July 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>
#include "../typecast.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <iostream>

#include <fl/util/math/linear_algebra.hpp>
#include <fl/filter/filter_interface.hpp>

#include <fl/filter/gaussian/gaussian_filter_linear.hpp>
#include <fl/model/process/linear_state_transition_model.hpp>
#include <fl/model/observation/linear_gaussian_observation_model.hpp>
#include <fl/model/observation/linear_decorrelated_gaussian_observation_model.hpp>

enum : signed int
{
    DecorrelatedGaussianModel,
    GaussianModel
};

static constexpr double epsilon = 0.1;

template <typename TestType>
class GaussianFilterLinearVsXTest
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

    typedef fl::IntegerTypeMap<
                fl::IntegerTypePair<
                    DecorrelatedGaussianModel,
                    fl::LinearDecorrelatedGaussianSensor<Obsrv, State>
                >,
                fl::IntegerTypePair<
                    GaussianModel,
                    fl::LinearGaussianSensor<Obsrv, State>
                >
            > SensorMap;


    typedef fl::LinearTransition<State, State, Input> LinearTransition;

    typedef typename SensorMap::template Select<
                Configuration::SelectedModel
            >::Type LinearSensor;

    typedef fl::GaussianFilter<
                LinearTransition, LinearSensor
            > KalmanFilter;

    typedef typename Configuration::template FilterDefinition<
                LinearTransition,
                LinearSensor
            > FilterDefinition;

    typedef typename FilterDefinition::Type Filter;

    GaussianFilterLinearVsXTest()
        : predict_steps_(200),
          predict_update_steps_(30)
    { }

    KalmanFilter create_kalman_filter() const
    {
        return KalmanFilter(
                LinearTransition(StateDim, InputDim),
                LinearSensor(ObsrvDim, StateDim));
    }

    Filter create_filter() const
    {
        return Configuration::create_filter(
                LinearTransition(StateDim, InputDim),
                LinearSensor(ObsrvDim, StateDim));
    }

    void setup_models(
        KalmanFilter& kalman_filter, Filter& other_filter, ModelSetup setup)
    {
        auto A = kalman_filter.process_model().create_dynamics_matrix();
        auto B = kalman_filter.process_model().create_input_matrix();
        auto Q = kalman_filter.process_model().create_noise_matrix();

        auto H = kalman_filter.obsrv_model().create_sensor_matrix();
        auto R = kalman_filter.obsrv_model().create_noise_matrix();

        Q.setZero();
        R.setZero();
        B.setZero();

        switch (setup)
        {
        case Random:
            A.diagonal().setRandom();
            H.diagonal().setRandom();
            Q.diagonal().setRandom(); Q *= Q.transpose().eval();
            R.diagonal().setRandom(); R *= R.transpose().eval();
            break;

        case Identity:
            A.setIdentity();
            H.setIdentity();
            Q.setIdentity();
            R.setIdentity();
            break;
        }

        kalman_filter.process_model().dynamics_matrix(A);
        kalman_filter.process_model().input_matrix(B);
        kalman_filter.process_model().noise_matrix(Q);
        kalman_filter.obsrv_model().sensor_matrix(H);
        kalman_filter.obsrv_model().noise_matrix(R);

        other_filter.process_model().dynamics_matrix(A);
        other_filter.process_model().input_matrix(B);
        other_filter.process_model().noise_matrix(Q);
        other_filter.obsrv_model().sensor_matrix(H);
        other_filter.obsrv_model().noise_matrix(R);
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

TYPED_TEST_CASE_P(GaussianFilterLinearVsXTest);

//TYPED_TEST_P(GaussianFilterLinearVsXTest, predict)
//{
//    typedef TestFixture This;

//    auto other_filter = This::create_filter();
//    auto kalman_filter = This::create_kalman_filter();

//    This::setup_models(kalman_filter, other_filter, This::Random);

//    auto belief_other = other_filter.create_belief();
//    auto belief_kf = kalman_filter.create_belief();

//    for (int i = 0; i < This::predict_steps_; ++i)
//    {
//        other_filter.predict(belief_other, This::zero_input(), belief_other);
//        kalman_filter.predict(belief_kf, This::zero_input(), belief_kf);

//        if (!fl::are_similar(belief_other.mean(), belief_kf.mean(), epsilon))
//        {
//            std::cout << "i = " << i << std::endl;
//            PV(belief_kf.mean());
//            PV(belief_other.mean());
//        }

//        if (!fl::are_similar(belief_other.covariance(), belief_kf.covariance(), epsilon))
//        {
//            std::cout << "i = " << i << std::endl;
//            PV(belief_kf.covariance());
//            PV(belief_other.covariance());
//        }

//        ASSERT_TRUE(
//            fl::are_similar(belief_other.mean(), belief_kf.mean(), epsilon));

//        ASSERT_TRUE(
//            fl::are_similar(belief_other.covariance(), belief_kf.covariance(), epsilon));
//    }

//    PV(belief_kf.mean());
//    PV(belief_other.mean());
//    PV(belief_kf.covariance());
//    PV(belief_other.covariance());
//}

TYPED_TEST_P(GaussianFilterLinearVsXTest, predict_and_update)
{
    typedef TestFixture This;

    auto other_filter = This::create_filter();
    auto kalman_filter = This::create_kalman_filter();

    This::setup_models(kalman_filter, other_filter, This::Random);

    auto belief_other = other_filter.create_belief();
    auto belief_kf = kalman_filter.create_belief();

    for (int i = 0; i < This::predict_update_steps_; ++i)
    {
        auto y = This::rand_obsrv();

        kalman_filter.predict(belief_kf, This::zero_input(), belief_kf);
        other_filter.predict(belief_other, This::zero_input(), belief_other);

//        if (!fl::are_similar(belief_other.mean(), belief_kf.mean(), epsilon))
//        {
//            std::cout << "predict i = " << i << std::endl;
//            PV(belief_kf.mean());
//            PV(belief_other.mean());
//        }

//        if (!fl::are_similar(belief_other.covariance(), belief_kf.covariance(), epsilon))
//        {
//            std::cout << "predict i = " << i << std::endl;
//            PV(belief_kf.covariance());
//            PV(belief_other.covariance());
//        }

        kalman_filter.update(belief_kf, y, belief_kf);
        other_filter.update(belief_other, y, belief_other);

//        if (!fl::are_similar(belief_other.mean(), belief_kf.mean(), epsilon))
//        {
//            std::cout << "update i = " << i << std::endl;
//            PV(belief_kf.mean());
//            PV(belief_other.mean());
//        }

//        if (!fl::are_similar(belief_other.covariance(), belief_kf.covariance(), epsilon))
//        {
//            std::cout << "update i = " << i << std::endl;
//            PV(belief_kf.covariance());
//            PV(belief_other.covariance());
//        }

        ASSERT_TRUE(
            fl::are_similar(belief_other.mean(), belief_kf.mean(), epsilon));

        ASSERT_TRUE(
            fl::are_similar(belief_other.covariance(), belief_kf.covariance(), epsilon));
    }

//    PV(belief_kf.mean());
//    PV(belief_other.mean());

//    PV(belief_kf.covariance());
//    PV(belief_other.covariance());

//    std::cout << other_filter.name() << std::endl;
}


REGISTER_TYPED_TEST_CASE_P(GaussianFilterLinearVsXTest,
                           predict_and_update);
