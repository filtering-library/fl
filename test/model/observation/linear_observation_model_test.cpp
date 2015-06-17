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

#include <Eigen/Dense>

#include <cmath>
#include <iostream>

#include <fl/util/types.hpp>
#include <fl/model/observation/linear_observation_model.hpp>

class LinearObservationModelTests:
    public testing::Test
{
public:
    enum: signed int
    {
        ObsrvDim = 20,
        StateDim = 10
    };

    typedef Eigen::Matrix<fl::FloatingPoint, StateDim, 1> State;
    typedef Eigen::Matrix<fl::FloatingPoint, ObsrvDim, 1> Obsrv;
    typedef fl::LinearObservationModel<Obsrv, State> LinearModel;

    typedef Eigen::Matrix<fl::FloatingPoint, Eigen::Dynamic, 1> StateX;
    typedef Eigen::Matrix<fl::FloatingPoint, Eigen::Dynamic, 1> ObsrvX;
    typedef fl::LinearObservationModel<ObsrvX, StateX> LinearModelX;

    typedef typename LinearModel::Noise Noise;
    typedef typename LinearModelX::Noise NoiseX;

    template <typename Model>
    void init_dimension_test(Model&& model)
    {
        EXPECT_EQ(model.obsrv_dimension(), ObsrvDim);
        EXPECT_EQ(model.noise_dimension(), ObsrvDim);
        EXPECT_EQ(model.state_dimension(), StateDim);

        EXPECT_EQ(model.sensor_matrix().rows(), ObsrvDim);
        EXPECT_EQ(model.sensor_matrix().cols(), StateDim);

        EXPECT_EQ(model.noise_matrix().rows(), ObsrvDim);
        EXPECT_EQ(model.noise_matrix().cols(), ObsrvDim);
    }

    template <typename Model>
    void init_sensor_matrix_value_test(Model&& model)
    {
        EXPECT_TRUE(model.sensor_matrix().isIdentity());
    }

    template <typename Model>
    void init_noise_matrix_value_test(Model&& model)
    {
        EXPECT_TRUE(model.noise_matrix().isIdentity());
    }

    template <typename Model>
    void sensor_matrix_value_test(Model&& model)
    {
        auto sensor_matrix = model.create_sensor_matrix();
        sensor_matrix.setRandom();
        model.sensor_matrix(sensor_matrix);
        EXPECT_TRUE(fl::are_similar(model.sensor_matrix(), sensor_matrix));
    }

    template <typename Model>
    void noise_matrix_value_test(Model&& model)
    {
        auto noise_matrix = model.create_noise_matrix();
        noise_matrix.setRandom();
        model.noise_matrix(noise_matrix);
        EXPECT_TRUE(fl::are_similar(model.noise_matrix(), noise_matrix));
    }

    template <typename Model>
    void expected_observation_test(Model&& model)
    {
        auto x = State(model.state_dimension());
        auto y = Obsrv(model.obsrv_dimension());

        x.setRandom();
        y.topRows(model.state_dimension()) = x;

        EXPECT_TRUE(fl::are_similar(model.expected_observation(x), y));
    }

    template <typename Model>
    void observation_with_zero_noise_test(Model&& model)
    {
        auto x = State(model.state_dimension());
        auto y = Obsrv(model.obsrv_dimension());
        auto v = Noise(model.noise_dimension());

        x.setRandom();
        v.setZero();
        y.topRows(model.state_dimension()) = x;

        EXPECT_TRUE(fl::are_similar(model.observation(x, v), y));
    }

    template <typename Model>
    void observation_test(Model&& model)
    {
        auto x = State(model.state_dimension());
        auto y = Obsrv(model.obsrv_dimension());
        auto v = Noise(model.noise_dimension());

        x.setRandom();
        v.setRandom();
        y.topRows(model.state_dimension()) = x;
        y += v;

        EXPECT_TRUE(fl::are_similar(model.observation(x, v), y));
    }
};

// == Initial dimension tests =============================================== //

TEST_F(LinearObservationModelTests, init_fixed_dimension)
{
    init_dimension_test(LinearModel());
}

TEST_F(LinearObservationModelTests, init_fixed_dimension_dynamic_constr)
{
    init_dimension_test(LinearModel(ObsrvDim, StateDim));
}

TEST_F(LinearObservationModelTests, init_dynamic_dimension)
{
    init_dimension_test(LinearModelX(ObsrvDim, StateDim));
}

// == Initial sensor matrix tests =========================================== //

TEST_F(LinearObservationModelTests, init_fixed_sensor_matrix_value)
{
    init_sensor_matrix_value_test(LinearModel());
}

TEST_F(LinearObservationModelTests, init_fixed_sensor_matrix_value_dynamic_constr)
{
    init_sensor_matrix_value_test(LinearModel(ObsrvDim, StateDim));
}

TEST_F(LinearObservationModelTests, init_dynamic_sensor_matrix_value)
{
    init_sensor_matrix_value_test(LinearModelX(ObsrvDim, StateDim));
}

// == Initial noise matrix tests ============================================ //

TEST_F(LinearObservationModelTests, init_fixed_noise_matrix_value)
{
    init_noise_matrix_value_test(LinearModel());
}

TEST_F(LinearObservationModelTests, init_fixed_noise_matrix_value_dynamic_constr)
{
    init_noise_matrix_value_test(LinearModel(ObsrvDim, StateDim));
}

TEST_F(LinearObservationModelTests, init_dynamic_noise_matrix_value)
{
    init_noise_matrix_value_test(LinearModelX(ObsrvDim, StateDim));
}

// == Sensor matrix tests =================================================== //

TEST_F(LinearObservationModelTests, sensor_matrix_fixed)
{
    sensor_matrix_value_test(LinearModel());
}

TEST_F(LinearObservationModelTests, sensor_matrix_fixed_dynamic_constr)
{
    sensor_matrix_value_test(LinearModel(ObsrvDim, StateDim));
}

TEST_F(LinearObservationModelTests, sensor_matrix_dynamic)
{
    sensor_matrix_value_test(LinearModelX(ObsrvDim, StateDim));
}

// == Noise matrix tests ==================================================== //

TEST_F(LinearObservationModelTests, noise_matrix_fixed)
{
    noise_matrix_value_test(LinearModel());
}

TEST_F(LinearObservationModelTests, noise_matrix_fixed_dynamic_constr)
{
    noise_matrix_value_test(LinearModel(ObsrvDim, StateDim));
}

TEST_F(LinearObservationModelTests, noise_matrix_dynamic)
{
    noise_matrix_value_test(LinearModelX(ObsrvDim, StateDim));
}

// == expected_observation ================================================== //

TEST_F(LinearObservationModelTests, expected_observation_fixed)
{
    expected_observation_test(LinearModel());
}

TEST_F(LinearObservationModelTests, expected_observation_dynamic_constr)
{
    expected_observation_test(LinearModel(ObsrvDim, StateDim));
}

TEST_F(LinearObservationModelTests, expected_observation_dynamic)
{
    expected_observation_test(LinearModelX(ObsrvDim, StateDim));
}

// == observation with zero noise =========================================== //

TEST_F(LinearObservationModelTests, observation_with_zero_noise_fixed)
{
    observation_with_zero_noise_test(LinearModel());
}

TEST_F(LinearObservationModelTests, observation_with_zero_noise_dynamic_constr)
{
    observation_with_zero_noise_test(LinearModel(ObsrvDim, StateDim));
}

TEST_F(LinearObservationModelTests, observation_with_zero_noise_dynamic)
{
    observation_with_zero_noise_test(LinearModelX(ObsrvDim, StateDim));
}

// == observation =========================================================== //

TEST_F(LinearObservationModelTests, observation_fixed)
{
    observation_test(LinearModel());
}

TEST_F(LinearObservationModelTests, observation_dynamic_constr)
{
    observation_test(LinearModel(ObsrvDim, StateDim));
}

TEST_F(LinearObservationModelTests, observation_dynamic)
{
    observation_test(LinearModelX(ObsrvDim, StateDim));
}

/// \todo missing probability and log_probability tests
