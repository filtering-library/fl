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
#include "../../typecast.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <iostream>

#include <fl/util/types.hpp>
#include <fl/model/process/linear_state_transition_model.hpp>

template <typename TestType>
class LinearStateTransitionModelTest:
    public testing::Test
{
public:
    enum: signed int
    {
        StateDim = TestType::Parameter::StateDim,
        InputDim = TestType::Parameter::InputDim,

        StateSize = fl::TestSize<StateDim, TestType>::Value,
        InputSize = fl::TestSize<InputDim, TestType>::Value
    };

    typedef Eigen::Matrix<fl::Real, StateDim, 1> State;
    typedef Eigen::Matrix<fl::Real, InputDim, 1> Input;
    typedef fl::LinearStateTransitionModel<State, Input> LinearModel;

    typedef typename LinearModel::Noise Noise;

    LinearStateTransitionModelTest()
        : model()
    { }

    void init_dimension_test()
    {
        EXPECT_EQ(model.state_dimension(), StateDim);
        EXPECT_EQ(model.noise_dimension(), StateDim);
        EXPECT_EQ(model.input_dimension(), InputDim);

        EXPECT_EQ(model.dynamics_matrix().rows(), StateDim);
        EXPECT_EQ(model.dynamics_matrix().cols(), StateDim);

        EXPECT_EQ(model.noise_matrix().rows(), StateDim);
        EXPECT_EQ(model.noise_matrix().cols(), StateDim);
    }

    void init_dynamics_matrix_value_test()
    {
        EXPECT_TRUE(model.dynamics_matrix().isIdentity());
    }

    void init_noise_matrix_value_test()
    {
        EXPECT_TRUE(model.noise_matrix().isIdentity());
    }

    void dynamics_matrix_value_test()
    {
        auto dynamics_matrix = model.dynamics_matrix();
        dynamics_matrix.setRandom();
        model.dynamics_matrix(dynamics_matrix);
        EXPECT_TRUE(fl::are_similar(model.dynamics_matrix(), dynamics_matrix));
    }

    void noise_matrix_value_test()
    {
        auto noise_matrix = model.noise_matrix();
        noise_matrix.setRandom();
        model.noise_matrix(noise_matrix);
        EXPECT_TRUE(fl::are_similar(model.noise_matrix(), noise_matrix));
    }

    void expected_state_test()
    {
        auto x = State(model.state_dimension());
        auto u = Input(model.input_dimension());

        x.setRandom();
        u.setRandom();

        EXPECT_TRUE(
            fl::are_similar(
                model.expected_state(x, u),
                x + model.input_matrix() * u));
    }

    void state_with_zero_noise_test()
    {
        auto x = State(model.state_dimension());
        auto u = Input(model.input_dimension());
        auto w = Noise(model.noise_dimension());

        x.setRandom();
        u.setRandom();
        w.setZero();

        EXPECT_TRUE(
            fl::are_similar(
                model.state(x, w, u),
                x + model.input_matrix() * u));
    }

    void state_test()
    {
        auto x = State(model.state_dimension());
        auto u = Input(model.input_dimension());
        auto w = Noise(model.noise_dimension());

        x.setRandom();
        u.setRandom();
        w.setRandom();

        EXPECT_TRUE(
            fl::are_similar(
                model.state(x, w, u),
                x + model.input_matrix() * u + w));
    }

protected:
    LinearModel model;
};

template <int StateDimension, int InputDimension>
struct Dimensions
{
    enum: signed int
    {
        StateDim = StateDimension,
        InputDim = InputDimension
    };
};

typedef ::testing::Types<
            fl::StaticTest<Dimensions<2, 1>>,
            fl::StaticTest<Dimensions<2, 2>>,
            fl::StaticTest<Dimensions<3, 3>>,
            fl::StaticTest<Dimensions<10, 10>>,
            fl::StaticTest<Dimensions<10, 20>>,
            fl::StaticTest<Dimensions<100, 10>>,
            fl::StaticTest<Dimensions<3, 100>>,
            fl::StaticTest<Dimensions<100, 100>>,
            fl::DynamicTest<Dimensions<2, 1>>,
            fl::DynamicTest<Dimensions<2, 2>>,
            fl::DynamicTest<Dimensions<3, 3>>,
            fl::DynamicTest<Dimensions<10, 10>>,
            fl::DynamicTest<Dimensions<10, 20>>,
            fl::DynamicTest<Dimensions<100, 10>>,
            fl::DynamicTest<Dimensions<3, 100>>,
            fl::DynamicTest<Dimensions<100, 100>>
        > TestTypes;

TYPED_TEST_CASE(LinearStateTransitionModelTest, TestTypes);

TYPED_TEST(LinearStateTransitionModelTest, init_dimension)
{
    TestFixture::init_dimension_test();
}

TYPED_TEST(LinearStateTransitionModelTest, init_dynamics_matrix_value)
{
    TestFixture::init_dynamics_matrix_value_test();
}
TYPED_TEST(LinearStateTransitionModelTest, init_noise_matrix_value)
{
    TestFixture::init_noise_matrix_value_test();
}

TYPED_TEST(LinearStateTransitionModelTest, dynamics_matrix)
{
    TestFixture::dynamics_matrix_value_test();
}

TYPED_TEST(LinearStateTransitionModelTest, noise_matrix)
{
    TestFixture::noise_matrix_value_test();
}

TYPED_TEST(LinearStateTransitionModelTest, expected_state)
{
    TestFixture::expected_state_test();
}

TYPED_TEST(LinearStateTransitionModelTest, state_with_zero_noise)
{
    TestFixture::state_with_zero_noise_test();
}

TYPED_TEST(LinearStateTransitionModelTest, state)
{
    TestFixture::state_test();
}


/// \todo missing probability and log_probability tests
