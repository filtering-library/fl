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
 * \file factorized_iid_sensor_test.cpp
 * \date January 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <iostream>

#include <fl/util/meta.hpp>
#include <fl/model/sensor/linear_sensor.hpp>
#include <fl/model/sensor/joint_sensor_iid.hpp>

constexpr static double sigma = 2.0;

class JointObservationModel_IID_Tests
    : public ::testing::Test
{
public:
    enum
    {
        StateDimension = 2,
        ObsrvDimension = 2,
        Factors = 3
    };

    //  local types of each marginal
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, StateDimension, 1> LocalState_F;
    typedef Eigen::Matrix<Scalar, ObsrvDimension, 1> LocalObservation_F;

    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> LocalState_D;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> LocalObservation_D;

    // local model
    typedef fl::LinearGaussianObservationModel<
                LocalObservation_F,
                LocalState_F
            > LocalObsrvModel_F;

    typedef fl::LinearGaussianObservationModel<
                LocalObservation_D,
                LocalState_D
            > LocalObsrvModel_D;

    typedef typename fl::Traits<LocalObsrvModel_F>::SecondMoment LocalCov_F;
    typedef typename fl::Traits<LocalObsrvModel_F>::SensorMatrix SensorMatrix_F;

    typedef typename fl::Traits<LocalObsrvModel_D>::SecondMoment LocalCov_D;
    typedef typename fl::Traits<LocalObsrvModel_D>::SensorMatrix SensorMatrix_D;

    // holistic model
    typedef fl::JointObservationModel<
                fl::MultipleOf<LocalObsrvModel_F, Factors>
            > ObservationModel_F;

    typedef fl::JointObservationModel<
                fl::MultipleOf<LocalObsrvModel_D, Eigen::Dynamic>
            > ObservationModel_D;

    typedef typename fl::Traits<ObservationModel_F>::State State_F;
    typedef typename fl::Traits<ObservationModel_F>::Noise Noise_F;
    typedef typename fl::Traits<ObservationModel_F>::Obsrv Obsrv_F;

    typedef typename fl::Traits<ObservationModel_D>::State State_D;
    typedef typename fl::Traits<ObservationModel_D>::Noise Noise_D;
    typedef typename fl::Traits<ObservationModel_D>::Obsrv Obsrv_D;
};

TEST_F(JointObservationModel_IID_Tests, dimensions_F)
{
    using namespace fl;

    EXPECT_EQ(Traits<ObservationModel_F>::StateDim, StateDimension);
    EXPECT_EQ(Traits<ObservationModel_F>::ObsrvDim, ObsrvDimension * Factors);
    EXPECT_EQ(Traits<ObservationModel_F>::NoiseDim, ObsrvDimension * Factors);

    auto model = ObservationModel_F(LocalObsrvModel_F());

    EXPECT_EQ(model.state_dimension(), StateDimension);
    EXPECT_EQ(model.obsrv_dimension(), ObsrvDimension * Factors);
    EXPECT_EQ(model.noise_dimension(), ObsrvDimension * Factors);

    model = ObservationModel_F(
                MultipleOf<LocalObsrvModel_F, Factors>(LocalObsrvModel_F()));

    EXPECT_EQ(model.state_dimension(), StateDimension);
    EXPECT_EQ(model.obsrv_dimension(), ObsrvDimension * Factors);
    EXPECT_EQ(model.noise_dimension(), ObsrvDimension * Factors);
}

TEST_F(JointObservationModel_IID_Tests, dimensions_D)
{
    using namespace fl;

    auto model = ObservationModel_D(
                     LocalObsrvModel_D(ObsrvDimension, StateDimension),
                     Factors);

    EXPECT_EQ(model.state_dimension(), StateDimension);
    EXPECT_EQ(model.obsrv_dimension(), ObsrvDimension * Factors);
    EXPECT_EQ(model.noise_dimension(), ObsrvDimension * Factors);

    model = ObservationModel_D(
        MultipleOf<LocalObsrvModel_D, -1>(
            LocalObsrvModel_D(ObsrvDimension, StateDimension), Factors));

    EXPECT_EQ(model.state_dimension(), StateDimension);
    EXPECT_EQ(model.obsrv_dimension(), ObsrvDimension * Factors);
    EXPECT_EQ(model.noise_dimension(), ObsrvDimension * Factors);
}

TEST_F(JointObservationModel_IID_Tests, predict_F)
{
    auto model = ObservationModel_F(
                    LocalObsrvModel_F(
                        LocalCov_F::Identity() * sigma * sigma));

    auto H = model.local_sensor().H();
    H.setIdentity();
    model.local_sensor().H(H);

    auto x = State_F::Ones();
    auto w = Noise_F::Ones();

    auto y = model.predict_obsrv(x, w, 1.);
    EXPECT_TRUE(fl::are_similar(y, Noise_F::Ones() + sigma * Noise_F::Ones()));
}

TEST_F(JointObservationModel_IID_Tests, predict_F_using_dynamic_interface)
{
    auto model = ObservationModel_F(
                    LocalObsrvModel_F(
                        LocalCov_F::Identity(ObsrvDimension,
                                             ObsrvDimension) * sigma*sigma,
                        ObsrvDimension,
                        StateDimension),
                    Factors);

    auto H = model.local_sensor().H();
    H.setIdentity();
    model.local_sensor().H(H);

    auto x = State_F::Ones(model.state_dimension(), 1);
    auto w = Noise_F::Ones(model.noise_dimension(), 1);

    auto y = model.predict_obsrv(x, w, 1.);
    EXPECT_TRUE(fl::are_similar(y, Noise_F::Ones() + sigma*Noise_F::Ones()));
}

TEST_F(JointObservationModel_IID_Tests, predict_D)
{
    auto model = ObservationModel_D(
                     LocalObsrvModel_D(
                         LocalCov_D::Identity(ObsrvDimension,
                                              ObsrvDimension) * sigma*sigma,
                         ObsrvDimension,
                         StateDimension),
                     Factors);

    auto H = model.local_sensor().H();
    H.setIdentity();
    model.local_sensor().H(H);

    auto x = State_D::Ones(model.state_dimension(), 1);
    auto w = Noise_D::Ones(model.noise_dimension(), 1);

    auto y = model.predict_obsrv(x, w, 1.);
    EXPECT_TRUE(fl::are_similar(y, Noise_F::Ones() + sigma*Noise_F::Ones()));
}

TEST_F(JointObservationModel_IID_Tests, predict_F_vs_D)
{
    auto model_D = ObservationModel_D(
                        LocalObsrvModel_D(
                           LocalCov_D::Identity(ObsrvDimension,
                                                ObsrvDimension) * sigma*sigma,
                           ObsrvDimension,
                           StateDimension),
                       Factors);

    auto model_F = ObservationModel_F(
                       LocalObsrvModel_F(
                            LocalCov_F::Identity() * sigma*sigma));

    auto H_D = model_D.local_sensor().H();
    H_D.setIdentity();
    model_D.local_sensor().H(H_D);

    auto H_F = model_F.local_sensor().H();
    H_F.setIdentity();
    model_F.local_sensor().H(H_D);

    auto x_D = State_D::Ones(model_D.state_dimension(), 1);
    auto w_D = Noise_D::Ones(model_D.noise_dimension(), 1);

    auto x_F = State_F::Ones();
    auto w_F = Noise_F::Ones();

    auto y_D = model_D.predict_obsrv(x_D, w_D, 1.);
    auto y_F = model_F.predict_obsrv(x_F, w_F, 1.);

    EXPECT_TRUE(fl::are_similar(y_D, y_F));

}

TEST_F(JointObservationModel_IID_Tests, predict_F_vs_D_using_dynamic_interface)
{
    auto model_D = ObservationModel_D(
                        LocalObsrvModel_D(
                            LocalCov_D::Identity(ObsrvDimension,
                                                 ObsrvDimension) * sigma*sigma,
                            ObsrvDimension,
                            StateDimension),
                        Factors);

    auto model_F = ObservationModel_F(
                        LocalObsrvModel_F(
                            LocalCov_F::Identity(ObsrvDimension,
                                                 ObsrvDimension) * sigma*sigma,
                            ObsrvDimension,
                            StateDimension),
                        Factors);

    auto H_D = model_D.local_sensor().H();
    H_D.setIdentity();
    model_D.local_sensor().H(H_D);

    auto H_F = model_F.local_sensor().H();
    H_F.setIdentity();
    model_F.local_sensor().H(H_F);

    auto x_D = State_D::Ones(model_D.state_dimension(), 1);
    auto w_D = Noise_D::Ones(model_D.noise_dimension(), 1);

    auto x_F = State_F::Ones(model_F.state_dimension(), 1);
    auto w_F = Noise_F::Ones(model_F.noise_dimension(), 1);

    auto y_D = model_D.predict_obsrv(x_D, w_D, 1.);
    auto y_F = model_F.predict_obsrv(x_F, w_F, 1.);

    EXPECT_TRUE(fl::are_similar(y_D, y_F));

}

