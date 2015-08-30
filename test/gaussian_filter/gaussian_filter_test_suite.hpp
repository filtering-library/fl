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

#include <fl/util/profiling.hpp>
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

    GaussianFilterTest()
        : predict_steps_(30),
          predict_update_steps_(30)
    { }

    struct ModelFactory
    {
        // Sizes will be 1 of the test type is static and it will fall back to
        // -1 for dynamic size tests. This may be used as an expansion factor
        // using fl::ExpandSizes<MySize, Sizes>::Value. If Sizes is equal to 1
        // (Static) then ExpandSizes will simply multiply MySizes by 1 and
        // everything is defined statically. On the other hand, if the Sizes
        // is -1, fl::ExpandSizes will fallback to -1 indicating that the test
        // type is dynamic.
        enum : signed int { Sizes = fl::TestSize<1, TestType>::Value };

        typedef fl::LinearStateTransitionModel<
                    State, Input
                > LinearStateTransition;

        typedef fl::LinearGaussianObservationModel<
                    Obsrv, State
                > LinearObservation;

        LinearStateTransition create_linear_state_model()
        {
            auto model = LinearStateTransition(StateDim, InputDim);

            auto A = model.create_dynamics_matrix();
            auto Q = model.create_noise_matrix();

            switch (setup)
            {
            case Random:
                A.setRandom();
                Q.setRandom();
                break;

            case Identity:
                A.setIdentity();
                Q.setIdentity();
                break;
            }

            model.dynamics_matrix(A);
            model.noise_matrix(Q);

            return model;
        }

        LinearObservation create_observation_model()
        {
            auto model = LinearObservation(ObsrvDim, StateDim);

            auto H = model.create_sensor_matrix();
            auto R = model.create_noise_matrix();

            switch (setup)
            {
            case Random:
                H.setRandom();
                R.setRandom();
                break;

            case Identity:
                H.setIdentity();
                R.setIdentity();
                break;
            }

            model.sensor_matrix(H);
            model.noise_matrix(R);

            return model;
        }

        ModelSetup setup;
    };

    typedef typename Configuration::template FilterDefinition<
                ModelFactory
            >::Type Filter;

    Filter create_filter(ModelSetup setup = Identity) const
    {
        return Configuration::create_filter(ModelFactory{setup});
    }

    typename fl::Traits<Filter>::Input zero_input(const Filter& filter)
    {
        return fl::Traits<Filter>::Input::Zero(
            filter.process_model().input_dimension());
    }

    typename fl::Traits<Filter>::Obsrv rand_obsrv(const Filter& filter)
    {
        return fl::Traits<Filter>::Obsrv::Random(
            filter.obsrv_model().obsrv_dimension());
    }

protected:
    int predict_steps_;
    int predict_update_steps_;
};

TYPED_TEST_CASE_P(GaussianFilterTest);

//TYPED_TEST_P(GaussianFilterTest, init_predict)
//{
//    typedef TestFixture This;

//    auto filter = This::create_filter();
//    auto belief = filter.create_belief();

//    EXPECT_TRUE(belief.mean().isZero());
//    EXPECT_TRUE(belief.covariance().isIdentity());

//    std::cout << filter.name() << std::endl;
//    std::cout << filter.description() << std::endl;

//    filter.predict(belief, This::zero_input(), belief);

//    auto Q = filter.process_model().noise_covariance();

//    EXPECT_TRUE(belief.mean().isZero());
//    EXPECT_TRUE(fl::are_similar(belief.covariance(), 2. * Q));
//}

TYPED_TEST_P(GaussianFilterTest, predict_then_update)
{
    typedef TestFixture This;

    auto filter = This::create_filter(This::Identity);

    PV(filter.name());

    auto belief = filter.create_belief();

    EXPECT_TRUE(belief.covariance().ldlt().isPositive());

    for (int i = 0; i < This::predict_update_steps_; ++i)
    {
        filter.predict(belief, This::zero_input(filter), belief);
        ASSERT_TRUE(belief.covariance().ldlt().isPositive());

        filter.update(belief, This::rand_obsrv(filter), belief);

        if (!belief.covariance().ldlt().isPositive())
        {
            PV(belief.mean());
            PV(belief.covariance());
        }

        ASSERT_TRUE(belief.covariance().ldlt().isPositive());
    }

    PV(belief.mean());
    PV(belief.covariance());
}

TYPED_TEST_P(GaussianFilterTest, predict_loop)
{
    typedef TestFixture This;

    auto filter = This::create_filter(This::Identity);
    auto belief = filter.create_belief();

    EXPECT_TRUE(belief.covariance().ldlt().isPositive());

    for (int i = 0; i < This::predict_steps_; ++i)
    {
        filter.predict(belief, This::zero_input(filter), belief);
    }

    EXPECT_TRUE(belief.covariance().ldlt().isPositive());
}

REGISTER_TYPED_TEST_CASE_P(GaussianFilterTest,
                           //init_predict,
                           predict_then_update,
                           predict_loop);


#endif
