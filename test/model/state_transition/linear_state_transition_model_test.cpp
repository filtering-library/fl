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
 * \file linear_state_transition_model_test.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <iostream>

#include <fl/util/types.hpp>
#include <fl/model/process/linear_process_model.hpp>

class LinearStateTransitionModelTests:
    public testing::Test
{
public:
    enum: signed int
    {
        StateDim = 10,
        InputDim = 10
    };

    typedef Eigen::Matrix<fl::FloatingPoint, StateDim, 1> State;
    typedef Eigen::Matrix<fl::FloatingPoint, InputDim, 1> Input;
    typedef fl::LinearStateTransitionModel<State, Input> LinearModel;

    typedef Eigen::Matrix<fl::FloatingPoint, Eigen::Dynamic, 1> StateX;
    typedef Eigen::Matrix<fl::FloatingPoint, Eigen::Dynamic, 1> InputX;
    typedef fl::LinearStateTransitionModel<StateX, InputX> LinearModelX;

    typedef typename LinearModel::Noise Noise;
    typedef typename LinearModelX::Noise NoiseX;

    template <typename Model>
    void init_dimension_test(Model&& model)
    {
        EXPECT_EQ(model.state_dimension(), StateDim);
        EXPECT_EQ(model.noise_dimension(), StateDim);
        EXPECT_EQ(model.input_dimension(), InputDim);

        EXPECT_EQ(model.dynamics_matrix().rows(), StateDim);
        EXPECT_EQ(model.dynamics_matrix().cols(), StateDim);

        EXPECT_EQ(model.noise_matrix().rows(), StateDim);
        EXPECT_EQ(model.noise_matrix().cols(), StateDim);
    }

    template <typename Model>
    void init_dynamics_matrix_value_test(Model&& model)
    {
        EXPECT_TRUE(model.dynamics_matrix().isIdentity());
    }

    template <typename Model>
    void init_noise_matrix_value_test(Model&& model)
    {
        EXPECT_TRUE(model.noise_matrix().isIdentity());
    }

    template <typename Model>
    void dynamics_matrix_value_test(Model&& model)
    {
        auto dynamics_matrix = model.dynamics_matrix();
        dynamics_matrix.setRandom();
        model.dynamics_matrix(dynamics_matrix);
        EXPECT_TRUE(fl::are_similar(model.dynamics_matrix(), dynamics_matrix));
    }

    template <typename Model>
    void noise_matrix_value_test(Model&& model)
    {
        auto noise_matrix = model.noise_matrix();
        noise_matrix.setRandom();
        model.noise_matrix(noise_matrix);
        EXPECT_TRUE(fl::are_similar(model.noise_matrix(), noise_matrix));
    }

    template <typename Model>
    void expected_state_test(Model&& model)
    {
        auto x = State(model.state_dimension());
        auto u = Input(model.input_dimension());

        x.setRandom();
        u.setRandom();

        EXPECT_TRUE(fl::are_similar(model.expected_state(x, u), x + u));
    }

    template <typename Model>
    void state_with_zero_noise_test(Model&& model)
    {
        auto x = State(model.state_dimension());
        auto u = Input(model.input_dimension());
        auto w = Noise(model.noise_dimension());

        x.setRandom();
        u.setRandom();
        w.setZero();

        EXPECT_TRUE(fl::are_similar(model.state(x, u, w), x + u));
    }

    template <typename Model>
    void state_test(Model&& model)
    {
        auto x = State(model.state_dimension());
        auto u = Input(model.input_dimension());
        auto w = Noise(model.noise_dimension());

        x.setRandom();
        u.setRandom();
        w.setRandom();

        EXPECT_TRUE(fl::are_similar(model.state(x, u, w), x + u + w));
    }
};

// == Initial dimension tests =============================================== //

TEST_F(LinearStateTransitionModelTests, init_fixed_dimension)
{
    init_dimension_test(LinearModel());
}

TEST_F(LinearStateTransitionModelTests, init_fixed_dimension_dynamic_constr)
{
    init_dimension_test(LinearModel(StateDim, InputDim));
}

TEST_F(LinearStateTransitionModelTests, init_dynamic_dimension)
{
    init_dimension_test(LinearModelX(StateDim, InputDim));
}

// == Initial sensor matrix tests =========================================== //

TEST_F(LinearStateTransitionModelTests, init_fixed_dynamics_matrix_value)
{
    init_dynamics_matrix_value_test(LinearModel());
}

TEST_F(LinearStateTransitionModelTests, init_fixed_dynamics_matrix_value_dynamic_constr)
{
    init_dynamics_matrix_value_test(LinearModel(StateDim, InputDim));
}

TEST_F(LinearStateTransitionModelTests, init_dynamic_dynamics_matrix_value)
{
    init_dynamics_matrix_value_test(LinearModelX(StateDim, InputDim));
}

// == Initial noise matrix tests ============================================ //

TEST_F(LinearStateTransitionModelTests, init_fixed_noise_matrix_value)
{
    init_noise_matrix_value_test(LinearModel());
}

TEST_F(LinearStateTransitionModelTests, init_fixed_noise_matrix_value_dynamic_constr)
{
    init_noise_matrix_value_test(LinearModel(StateDim, InputDim));
}

TEST_F(LinearStateTransitionModelTests, init_dynamic_noise_matrix_value)
{
    init_noise_matrix_value_test(LinearModelX(StateDim, InputDim));
}

// == Sensor matrix tests =================================================== //

TEST_F(LinearStateTransitionModelTests, dynamics_matrix_fixed)
{
    dynamics_matrix_value_test(LinearModel());
}

TEST_F(LinearStateTransitionModelTests, dynamics_matrix_fixed_dynamic_constr)
{
    dynamics_matrix_value_test(LinearModel(StateDim, InputDim));
}

TEST_F(LinearStateTransitionModelTests, dynamics_matrix_dynamic)
{
    dynamics_matrix_value_test(LinearModelX(StateDim, InputDim));
}

// == Noise matrix tests ==================================================== //

TEST_F(LinearStateTransitionModelTests, noise_matrix_fixed)
{
    noise_matrix_value_test(LinearModel());
}

TEST_F(LinearStateTransitionModelTests, noise_matrix_fixed_dynamic_constr)
{
    noise_matrix_value_test(LinearModel(StateDim, InputDim));
}

TEST_F(LinearStateTransitionModelTests, noise_matrix_dynamic)
{
    noise_matrix_value_test(LinearModelX(StateDim, InputDim));
}

// == expected_observation ================================================== //

TEST_F(LinearStateTransitionModelTests, expected_state_fixed)
{
    expected_state_test(LinearModel());
}

TEST_F(LinearStateTransitionModelTests, expected_state_dynamic_constr)
{
    expected_state_test(LinearModel(StateDim, InputDim));
}

TEST_F(LinearStateTransitionModelTests, expected_state_dynamic)
{
    expected_state_test(LinearModelX(StateDim, InputDim));
}

// == observation with zero noise =========================================== //

TEST_F(LinearStateTransitionModelTests, state_with_zero_noise_fixed)
{
    state_with_zero_noise_test(LinearModel());
}

TEST_F(LinearStateTransitionModelTests, state_with_zero_noise_dynamic_constr)
{
    state_with_zero_noise_test(LinearModel(StateDim, InputDim));
}

TEST_F(LinearStateTransitionModelTests, state_with_zero_noise_dynamic)
{
    state_with_zero_noise_test(LinearModelX(StateDim, InputDim));
}

// == observation =========================================================== //

TEST_F(LinearStateTransitionModelTests, state_fixed)
{
    state_test(LinearModel());
}

TEST_F(LinearStateTransitionModelTests, state_dynamic_constr)
{
    state_test(LinearModel(StateDim, InputDim));
}

TEST_F(LinearStateTransitionModelTests, state_dynamic)
{
    state_test(LinearModelX(StateDim, InputDim));
}

/// \todo missing probability and log_probability tests
