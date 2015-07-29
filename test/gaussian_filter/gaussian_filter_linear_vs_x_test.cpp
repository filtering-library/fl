/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Max-Planck-Institute for Intelligent Systems,
 *                     University of Southern California
 *    Jan Issac (jan.issac@gmail.com)
 *    Manuel Wuthrich (manuel.wuthrich@gmail.com)
 *
 *
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

/**
 * @date 2014
 * @author Jan Issac (jan.issac@gmail.com)
 * Max-Planck-Institute for Intelligent Systems,
 * University of Southern California
 */


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

    typedef fl::LinearStateTransitionModel<State, Input> LinearStateTransition;
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
