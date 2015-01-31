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
 * \file factorized_iid_observation_model_test.cpp
 * \date January 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <iostream>

#include <fl/model/observation/linear_observation_model.hpp>
#include <fl/model/observation/factorized_iid_observation_model.hpp>

constexpr static double sigma = 2.0;

class FactorizedIIDObservationModelTests
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
    typedef fl::FactorizedIIDObservationModel<
                LocalObsrvModel_F,
                Factors
            > ObservationModel_F;

    typedef fl::FactorizedIIDObservationModel<
                LocalObsrvModel_D,
                Eigen::Dynamic
            > ObservationModel_D;

    typedef typename fl::Traits<ObservationModel_F>::State State_F;
    typedef typename fl::Traits<ObservationModel_F>::Noise Noise_F;
    typedef typename fl::Traits<ObservationModel_F>::Observation Observation_F;

    typedef typename fl::Traits<ObservationModel_D>::State State_D;
    typedef typename fl::Traits<ObservationModel_D>::Noise Noise_D;
    typedef typename fl::Traits<ObservationModel_D>::Observation Observation_D;
};

TEST_F(FactorizedIIDObservationModelTests, predict_F)
{
    auto model = ObservationModel_F(
                    std::make_shared<LocalObsrvModel_F>(
                        LocalCov_F::Identity() * sigma*sigma));

    auto H = model.local_observation_model()->H();
    H.setIdentity();
    model.local_observation_model()->H(H);

    auto x = State_F::Ones();
    auto w = Noise_F::Ones();

    auto y = model.predict_observation(x, w, 1.);
    EXPECT_TRUE(y.isApprox(Noise_F::Ones() + sigma*Noise_F::Ones()));
}

TEST_F(FactorizedIIDObservationModelTests, predict_F_using_dynamic_interface)
{
    auto model = ObservationModel_F(
                    std::make_shared<LocalObsrvModel_F>(
                        LocalCov_F::Identity(ObsrvDimension,
                                             ObsrvDimension) * sigma*sigma,
                        ObsrvDimension,
                        StateDimension),
                    Factors);

    auto H = model.local_observation_model()->H();
    H.setIdentity();
    model.local_observation_model()->H(H);

    auto x = State_F::Ones(model.state_dimension(), 1);
    auto w = Noise_F::Ones(model.noise_dimension(), 1);

    auto y = model.predict_observation(x, w, 1.);
    EXPECT_TRUE(y.isApprox(Noise_F::Ones() + sigma*Noise_F::Ones()));
}


TEST_F(FactorizedIIDObservationModelTests, predict_D)
{
    auto model = ObservationModel_D(
                     std::make_shared<LocalObsrvModel_D>(
                         LocalCov_D::Identity(ObsrvDimension,
                                              ObsrvDimension) * sigma*sigma,
                         ObsrvDimension,
                         StateDimension),
                     Factors);

    auto H = model.local_observation_model()->H();
    H.setIdentity();
    model.local_observation_model()->H(H);

    auto x = State_D::Ones(model.state_dimension(), 1);
    auto w = Noise_D::Ones(model.noise_dimension(), 1);

    auto y = model.predict_observation(x, w, 1.);
    EXPECT_TRUE(y.isApprox(Noise_F::Ones() + sigma*Noise_F::Ones()));
}

TEST_F(FactorizedIIDObservationModelTests, predict_F_vs_D)
{
    auto model_D = ObservationModel_D(
                        std::make_shared<LocalObsrvModel_D>(
                           LocalCov_D::Identity(ObsrvDimension,
                                                ObsrvDimension) * sigma*sigma,
                           ObsrvDimension,
                           StateDimension),
                       Factors);

    auto model_F = ObservationModel_F(
                       std::make_shared<LocalObsrvModel_F>(
                            LocalCov_F::Identity() * sigma*sigma));

    auto H_D = model_D.local_observation_model()->H();
    H_D.setIdentity();
    model_D.local_observation_model()->H(H_D);

    auto H_F = model_F.local_observation_model()->H();
    H_F.setIdentity();
    model_F.local_observation_model()->H(H_D);

    auto x_D = State_D::Ones(model_D.state_dimension(), 1);
    auto w_D = Noise_D::Ones(model_D.noise_dimension(), 1);

    auto x_F = State_F::Ones();
    auto w_F = Noise_F::Ones();

    auto y_D = model_D.predict_observation(x_D, w_D, 1.);
    auto y_F = model_F.predict_observation(x_F, w_F, 1.);

    EXPECT_TRUE(y_D.isApprox(y_F));

}

TEST_F(FactorizedIIDObservationModelTests, predict_F_vs_D_using_dynamic_interface)
{
    auto model_D = ObservationModel_D(
                        std::make_shared<LocalObsrvModel_D>(
                            LocalCov_D::Identity(ObsrvDimension,
                                                 ObsrvDimension) * sigma*sigma,
                            ObsrvDimension,
                            StateDimension),
                        Factors);

    auto model_F = ObservationModel_F(
                        std::make_shared<LocalObsrvModel_F>(
                            LocalCov_F::Identity(ObsrvDimension,
                                                 ObsrvDimension) * sigma*sigma,
                            ObsrvDimension,
                            StateDimension),
                        Factors);

    auto H_D = model_D.local_observation_model()->H();
    H_D.setIdentity();
    model_D.local_observation_model()->H(H_D);

    auto H_F = model_F.local_observation_model()->H();
    H_F.setIdentity();
    model_F.local_observation_model()->H(H_D);

    auto x_D = State_D::Ones(model_D.state_dimension(), 1);
    auto w_D = Noise_D::Ones(model_D.noise_dimension(), 1);

    auto x_F = State_F::Ones(model_F.state_dimension(), 1);
    auto w_F = Noise_F::Ones(model_F.state_dimension(), 1);

    auto y_D = model_D.predict_observation(x_D, w_D, 1.);
    auto y_F = model_F.predict_observation(x_F, w_F, 1.);

    EXPECT_TRUE(y_D.isApprox(y_F));

}
