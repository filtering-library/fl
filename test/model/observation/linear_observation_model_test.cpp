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
 * \file linear_observation_model_test.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>
#include "../../typecast.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <iostream>

#include <fl/util/types.hpp>
#include <fl/model/observation/linear_observation_model.hpp>

template <typename TestType>
class LinearObservationModelTest:
    public testing::Test
{
public:
    enum: signed int
    {
        StateDim = 10,
        ObsrvDim = 20,

        StateSize = fl::TestSize<StateDim, TestType>::Value,
        ObsrvSize = fl::TestSize<ObsrvDim, TestType>::Value
    };

    typedef Eigen::Matrix<fl::Real, StateSize, 1> State;
    typedef Eigen::Matrix<fl::Real, ObsrvSize, 1> Obsrv;
    typedef fl::LinearObservationModel<Obsrv, State> LinearModel;

    typedef typename LinearModel::Noise Noise;

    LinearObservationModelTest()
        : model(LinearModel(ObsrvDim, StateDim))
    { }

    void init_dimension_test()
    {
        EXPECT_EQ(model.obsrv_dimension(), ObsrvDim);
        EXPECT_EQ(model.noise_dimension(), ObsrvDim);
        EXPECT_EQ(model.state_dimension(), StateDim);

        EXPECT_EQ(model.sensor_matrix().rows(), ObsrvDim);
        EXPECT_EQ(model.sensor_matrix().cols(), StateDim);

        EXPECT_EQ(model.noise_matrix().rows(), ObsrvDim);
        EXPECT_EQ(model.noise_matrix().cols(), ObsrvDim);
    }

    void init_sensor_matrix_value_test()
    {
        EXPECT_TRUE(model.sensor_matrix().isIdentity());
    }

    void init_noise_matrix_value_test()
    {
        EXPECT_TRUE(model.noise_matrix().isIdentity());
    }

    void sensor_matrix_value_test()
    {
        auto sensor_matrix = model.create_sensor_matrix();
        sensor_matrix.setRandom();
        model.sensor_matrix(sensor_matrix);
        EXPECT_TRUE(fl::are_similar(model.sensor_matrix(), sensor_matrix));
    }

    void noise_matrix_value_test()
    {
        auto noise_matrix = model.create_noise_matrix();
        noise_matrix.setRandom();
        model.noise_matrix(noise_matrix);
        EXPECT_TRUE(fl::are_similar(model.noise_matrix(), noise_matrix));
    }

    void expected_observation_test()
    {
        auto x = State(model.state_dimension());
        auto y = Obsrv(model.obsrv_dimension());

        x.setRandom();
        y.setZero();
        y.topRows(model.state_dimension()) = x;

        EXPECT_TRUE(fl::are_similar(model.expected_observation(x), y));
    }

    void observation_with_zero_noise_test()
    {
        auto x = State(model.state_dimension());
        auto y = Obsrv(model.obsrv_dimension());
        auto v = Noise(model.noise_dimension());

        x.setRandom();
        v.setZero();
        y.setZero();
        y.topRows(model.state_dimension()) = x;

        EXPECT_TRUE(fl::are_similar(model.observation(x, v), y));
    }

    void observation_test()
    {
        auto x = State(model.state_dimension());
        auto y = Obsrv(model.obsrv_dimension());
        auto v = Noise(model.noise_dimension());

        x.setRandom();
        v.setRandom();
        y.setZero();
        y.topRows(model.state_dimension()) = x;
        y += v;

        EXPECT_TRUE(fl::are_similar(model.observation(x, v), y));
    }

protected:
    LinearModel model;
};

typedef ::testing::Types<
            fl::StaticTest,
            fl::DynamicTest
        > TestTypes;

TYPED_TEST_CASE(LinearObservationModelTest, TestTypes);

TYPED_TEST(LinearObservationModelTest, init_dimension)
{
    TestFixture::init_dimension_test();
}

TYPED_TEST(LinearObservationModelTest, init_sensor_matrix_value)
{
    TestFixture::init_sensor_matrix_value_test();
}

TYPED_TEST(LinearObservationModelTest, init_noise_matrix_value)
{
    TestFixture::init_noise_matrix_value_test();
}

TYPED_TEST(LinearObservationModelTest, sensor_matrix)
{
    TestFixture::sensor_matrix_value_test();
}

TYPED_TEST(LinearObservationModelTest, noise_matrix)
{
    TestFixture::noise_matrix_value_test();
}

TYPED_TEST(LinearObservationModelTest, expected_observation)
{
    TestFixture::expected_observation_test();
}

TYPED_TEST(LinearObservationModelTest, observation_with_zero_noise)
{
    TestFixture::observation_with_zero_noise_test();
}

TYPED_TEST(LinearObservationModelTest, observation)
{
    TestFixture::observation_test();
}

/// \todo missing probability and log_probability tests
