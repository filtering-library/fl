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

    template <typename Model>
    void init_dimension_test(Model&& model)
    {
        EXPECT_EQ(model.obsrv_dimension(), ObsrvDim);
        EXPECT_EQ(model.noise_dimension(), ObsrvDim);
        EXPECT_EQ(model.state_dimension(), StateDim);
        EXPECT_TRUE(model.sensor_matrix().isIdentity());
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


//TEST_F(LinearObservationModelTests, predict_fixedsize_with_zero_noise)
//{
//    typedef Eigen::Matrix<fl::FloatingPoint, 10, 1> State;
//    typedef Eigen::Matrix<fl::FloatingPoint, 20, 1> Obsrv;

//    const int ObsrvDim = Obsrv::SizeAtCompileTime;
//    const int StateDim = State::SizeAtCompileTime;

//    typedef fl::LinearObservationModel<Obsrv, State> LinearModel;

//    State state = State::Random(StateDim, 1);
//    Obsrv observation = Obsrv::Random(ObsrvDim, 1);

//    LinearModel::Noise noise = LinearModel::Noise::Zero(ObsrvDim, 1);

//    LinearModel model;

//    model.noise_matrix(
//        LinearModel::NoiseMatrix::Identity(ObsrvDim, ObsrvDim) * 5.5465);

//    EXPECT_FALSE(fl::are_similar(model.map_standard_normal(noise), observation));
//    model.condition(state);
//    EXPECT_FALSE(fl::are_similar(model.map_standard_normal(noise), observation));
//}

//TEST_F(LinearObservationModelTests, predict_dynamic_with_zero_noise)
//{
//    const int ObsrvDim = 20;
//    const int StateDim = 10;
//    typedef Eigen::Matrix<fl::FloatingPoint, -1, 1> State;
//    typedef Eigen::Matrix<fl::FloatingPoint, -1, 1> Obsrv;
//    typedef fl::LinearObservationModel<Obsrv, State> LinearModel;

//    State state = State::Random(StateDim, 1);
//    Obsrv observation = Obsrv::Random(ObsrvDim, 1);
//    LinearModel::Noise noise = LinearModel::Noise::Zero(ObsrvDim, 1);
//    LinearModel::SecondMoment cov = LinearModel::SecondMoment::Identity(ObsrvDim, ObsrvDim) * 5.5465;
//    LinearModel model(cov, ObsrvDim, StateDim);

//    EXPECT_TRUE(model.map_standard_normal(noise).isZero());

//    EXPECT_FALSE(fl::are_similar(model.map_standard_normal(noise), observation));
//    model.condition(state);
//    EXPECT_FALSE(fl::are_similar(model.map_standard_normal(noise), observation));
//}

//TEST_F(LinearObservationModelTests, sensor_matrix)
//{
//    const int ObsrvDim = 20;
//    const int StateDim = 10;
//    typedef Eigen::Matrix<fl::FloatingPoint, -1, 1> State;
//    typedef Eigen::Matrix<fl::FloatingPoint, -1, 1> Obsrv;
//    typedef fl::LinearObservationModel<Obsrv, State> LinearModel;

//    State state = State::Random(StateDim, 1);
//    Obsrv observation = Obsrv::Zero(ObsrvDim, 1);
//    LinearModel::Noise noise = LinearModel::Noise::Random(ObsrvDim, 1);
//    LinearModel::SecondMoment cov = LinearModel::SecondMoment::Identity(ObsrvDim, ObsrvDim);
//    LinearModel::SensorMatrix H = LinearModel::SecondMoment::Ones(ObsrvDim, StateDim);
//    LinearModel model(cov, ObsrvDim, StateDim);

//    observation.topRows(StateDim) = state;

//    EXPECT_TRUE(fl::are_similar(model.map_standard_normal(noise), noise));
//    EXPECT_FALSE(fl::are_similar(model.map_standard_normal(noise), observation));

//    model.condition(state);

//    EXPECT_TRUE(
//        fl::are_similar(model.map_standard_normal(noise), H * state + noise));
//    EXPECT_FALSE(
//        fl::are_similar(model.map_standard_normal(noise), observation));

//    H = LinearModel::SecondMoment::Zero(ObsrvDim, StateDim);
//    H.block(0, 0, StateDim, StateDim)
//            = Eigen::MatrixXd::Identity(StateDim, StateDim);

//    model.H(H);
//    model.condition(state);

//    EXPECT_TRUE(
//        fl::are_similar(model.map_standard_normal(noise), observation + noise));
//}

