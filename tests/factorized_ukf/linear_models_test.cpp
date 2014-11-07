/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Max-Planck-Institute for Intelligent Systems,
 *                     University of Southern California
 *    Jan Issac (jan.issac@gmail.com)
 *    Manuel Wuthrich (manuel.wuthrich@gmail.com)
 *
 *  All rights reserved.
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
 * @author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * Max-Planck-Institute for Intelligent Systems, University of Southern California
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <iostream>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <fast_filtering/models/process_models/linear_process_model.hpp>
#include <fast_filtering/models/observation_models/linear_observation_model.hpp>
#include <fast_filtering/filters/deterministic/factorized_unscented_kalman_filter.hpp>

TEST(FUKFAndLinearModels, init_fixedsize_predict)
{
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, 10, 1> State_a;
    typedef Eigen::Matrix<Scalar, 1, 1> State_b;
    typedef Eigen::Matrix<Scalar, 20, 1> Observation;

    typedef ff::LinearGaussianProcessModel<State_a> ProcessModel_a;
    typedef ff::LinearGaussianProcessModel<State_b> ProcessModel_b;
    typedef ff::FactorizedLinearGaussianObservationModel<
                    Observation,
                    State_a,
                    State_b> ObservationModel;

    typedef ff::FactorizedUnscentedKalmanFilter<ProcessModel_a,
                                                ProcessModel_b,
                                                ObservationModel> Filter;

    ProcessModel_a::Operator Q_a = ProcessModel_a::Operator::Identity();
    ProcessModel_b::Operator Q_b = ProcessModel_b::Operator::Identity();
    ObservationModel::Operator R = ObservationModel::Operator::Identity();

    Filter::CohesiveStateProcessModelPtr process_model =
            boost::make_shared<ProcessModel_a>(Q_a);
    Filter::FactorizedStateProcessModelPtr process_model_b =
            boost::make_shared<ProcessModel_b>(Q_b);
    Filter::ObservationModelPtr observation_model =
            boost::make_shared<ObservationModel>(R);

    Filter filter(process_model, process_model_b, observation_model);

    Filter::StateDistribution state_dist;
    state_dist.initialize(State_a::Zero(),
                          1,
                          State_b::Zero(),
                          1.0,
                          1.0);

    EXPECT_TRUE(state_dist.mean_a.isZero());
    EXPECT_TRUE(state_dist.cov_aa.isIdentity());
    EXPECT_TRUE(state_dist.joint_partitions[0].mean_b.isZero());
    EXPECT_TRUE(state_dist.joint_partitions[0].cov_bb.isIdentity());
    filter.Predict(state_dist, 1.0, state_dist);
    EXPECT_TRUE(state_dist.mean_a.isZero());
    EXPECT_TRUE(state_dist.cov_aa.isApprox(2. * Q_a));
    EXPECT_TRUE(state_dist.joint_partitions[0].mean_b.isZero());
    EXPECT_TRUE(state_dist.joint_partitions[0].cov_bb.isApprox(2. * Q_b));
}

TEST(FUKFAndLinearModels, init_dynamicsize_predict)
{
    const size_t dim_state_a = 10;
    const size_t dim_state_b = 1;
    const size_t dim_observation = 20;


    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> State_a;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> State_b;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Observation;

    typedef ff::LinearGaussianProcessModel<State_a> ProcessModel_a;
    typedef ff::LinearGaussianProcessModel<State_b> ProcessModel_b;
    typedef ff::FactorizedLinearGaussianObservationModel<
                    Observation,
                    State_a,
                    State_b> ObservationModel;

    typedef ff::FactorizedUnscentedKalmanFilter<ProcessModel_a,
                                                ProcessModel_b,
                                                ObservationModel> Filter;

    ProcessModel_a::Operator Q_a =
            ProcessModel_a::Operator::Identity(dim_state_a,
                                               dim_state_a);
    ProcessModel_b::Operator Q_b =
            ProcessModel_b::Operator::Identity(dim_state_b,
                                               dim_state_b);
    ObservationModel::Operator R =
            ObservationModel::Operator::Identity(dim_observation,
                                                 dim_observation);

    Filter::CohesiveStateProcessModelPtr process_model_a =
            boost::make_shared<ProcessModel_a>(Q_a, dim_state_a);
    Filter::FactorizedStateProcessModelPtr process_model_b =
            boost::make_shared<ProcessModel_b>(Q_b, dim_state_b);
    Filter::ObservationModelPtr observation_model =
            boost::make_shared<ObservationModel>(R,
                                                 dim_observation,
                                                 dim_state_a,
                                                 dim_state_b);

    Filter filter(process_model_a, process_model_b, observation_model);

    Filter::StateDistribution state_dist;
    state_dist.initialize(State_a::Zero(dim_state_a, 1),
                          1,
                          State_b::Zero(dim_state_b, 1),
                          1.0,
                          1.0);

    EXPECT_TRUE(state_dist.mean_a.isZero());
    EXPECT_TRUE(state_dist.cov_aa.isIdentity());
    EXPECT_TRUE(state_dist.joint_partitions[0].mean_b.isZero());
    EXPECT_TRUE(state_dist.joint_partitions[0].cov_bb.isIdentity());
    filter.Predict(state_dist, 1.0, state_dist);
    EXPECT_TRUE(state_dist.mean_a.isZero());
    EXPECT_TRUE(state_dist.cov_aa.isApprox(2. * Q_a));
    EXPECT_TRUE(state_dist.joint_partitions[0].mean_b.isZero());
    EXPECT_TRUE(state_dist.joint_partitions[0].cov_bb.isApprox(2. * Q_b));
}


TEST(FUKFAndLinearModels, fixedsize_predict_update)
{
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, 10, 1> State_a;
    typedef Eigen::Matrix<Scalar, 2, 1> State_b;
    typedef Eigen::Matrix<Scalar, 2, 1> Observation;
    typedef Eigen::Matrix<
            Scalar,
            2 * Observation::RowsAtCompileTime,
            1> JointObservation;

    typedef ff::LinearGaussianProcessModel<State_a> ProcessModel_a;
    typedef ff::LinearGaussianProcessModel<State_b> ProcessModel_b;
    typedef ff::FactorizedLinearGaussianObservationModel<
                    Observation,
                    State_a,
                    State_b> ObservationModel;

    typedef ff::FactorizedUnscentedKalmanFilter<ProcessModel_a,
                                                ProcessModel_b,
                                                ObservationModel> Filter;

    ProcessModel_a::Operator Q_a =
            ProcessModel_a::Operator::Identity();
    ProcessModel_b::Operator Q_b =
            ProcessModel_b::Operator::Identity();


    ProcessModel_a::DynamicsMatrix A_a =
            ProcessModel_a::DynamicsMatrix::Identity();
    ProcessModel_b::DynamicsMatrix A_b =
            ProcessModel_b::DynamicsMatrix::Identity();

    ObservationModel::Operator R =
            ObservationModel::Operator::Identity();

    ObservationModel::SensorMatrix_a H_a =
            ObservationModel::SensorMatrix_a::Random();
    ObservationModel::SensorMatrix_b H_b =
            ObservationModel::SensorMatrix_b::Random();

    Q_a *= Q_a.transpose() * 0.05;
    Q_b *= Q_b.transpose() * 0.055;
    R *= 0.0005;
    A_a *= 0.005;
    A_b *= 0.005;

    Filter::CohesiveStateProcessModelPtr process_model_a =
            boost::make_shared<ProcessModel_a>(Q_a);
    Filter::FactorizedStateProcessModelPtr process_model_b =
            boost::make_shared<ProcessModel_b>(Q_b);
    Filter::ObservationModelPtr observation_model =
            boost::make_shared<ObservationModel>(R);

    process_model_a->A(A_a);
    process_model_b->A(A_b);
    observation_model->H_a(H_a);
    observation_model->H_b(H_b);

    Filter filter(process_model_a, process_model_b, observation_model);

    Filter::StateDistribution state_dist;
    state_dist.initialize(State_a::Zero(),
                          2,
                          State_b::Zero(),
                          1.0,
                          1.0);

    EXPECT_TRUE(state_dist.cov_aa.ldlt().isPositive());
    EXPECT_TRUE(state_dist.joint_partitions[0].cov_bb.ldlt().isPositive());

    for (size_t i = 0; i < 20000; ++i)
    {
        filter.Predict(state_dist, 1.0, state_dist);

//        std::cout << "predict \n" <<  state_dist.cov_aa << std::endl;
//        std::cout << "predict \n" <<  state_dist.joint_partitions[0].cov_bb << std::endl;

        EXPECT_TRUE(state_dist.cov_aa.ldlt().isPositive());
        EXPECT_TRUE(state_dist.joint_partitions[0].cov_bb.ldlt().isPositive());

        JointObservation y = JointObservation::Random();
        filter.Update(state_dist, y, state_dist);

//        std::cout << "update \n" << state_dist.cov_aa << std::endl;
//        std::cout << "update \n" << state_dist.joint_partitions[0].cov_bb << std::endl;

        EXPECT_TRUE(state_dist.cov_aa.ldlt().isPositive());
        EXPECT_TRUE(state_dist.joint_partitions[0].cov_bb.ldlt().isPositive());

        if (!state_dist.cov_aa.ldlt().isPositive())
        {
            ADD_FAILURE();
            break;
        }
    }
}


TEST(FUKFAndLinearModels, dynamicsize_predict_update)
{
    const size_t dim_state_a = 10;
    const size_t dim_state_b = 1;
    const size_t dim_observation = 1;

    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> State_a;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> State_b;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Observation;

    typedef ff::LinearGaussianProcessModel<State_a> ProcessModel_a;
    typedef ff::LinearGaussianProcessModel<State_b> ProcessModel_b;
    typedef ff::FactorizedLinearGaussianObservationModel<
                    Observation,
                    State_a,
                    State_b> ObservationModel;

    typedef ff::FactorizedUnscentedKalmanFilter<ProcessModel_a,
                                                ProcessModel_b,
                                                ObservationModel> Filter;

    ProcessModel_a::Operator Q_a =
            ProcessModel_a::Operator::Random(dim_state_a, dim_state_a);
    ProcessModel_b::Operator Q_b =
            ProcessModel_b::Operator::Random(dim_state_b, dim_state_b);

    ProcessModel_a::DynamicsMatrix A_a =
            ProcessModel_a::DynamicsMatrix::Identity(dim_state_a,
                                                   dim_state_a);
    ProcessModel_b::DynamicsMatrix A_b =
            ProcessModel_b::DynamicsMatrix::Identity(dim_state_b,
                                                   dim_state_b);

    ObservationModel::Operator R =
            ObservationModel::Operator::Identity(dim_observation,
                                                 dim_observation);

    ObservationModel::SensorMatrix_a H_a =
            ObservationModel::SensorMatrix_a::Random(dim_observation,
                                                     dim_state_a);
    ObservationModel::SensorMatrix_b H_b =
            ObservationModel::SensorMatrix_b::Random(dim_observation,
                                                     dim_state_b);

    Q_a *= Q_a.transpose() * 0.05;
    Q_b *= Q_b.transpose() * 0.055;
    R *= 0.0005;
    A_a *= 0.005;
    A_b *= 0.005;

    Filter::CohesiveStateProcessModelPtr process_model_a =
            boost::make_shared<ProcessModel_a>(Q_a, dim_state_a);
    Filter::FactorizedStateProcessModelPtr process_model_b =
            boost::make_shared<ProcessModel_b>(Q_b, dim_state_b);
    Filter::ObservationModelPtr observation_model =
            boost::make_shared<ObservationModel>(R,
                                                 dim_observation,
                                                 dim_state_a,
                                                 dim_state_b);

    process_model_a->A(A_a);
    process_model_b->A(A_b);
    observation_model->H_a(H_a);
    observation_model->H_b(H_b);

    Filter filter(process_model_a, process_model_b, observation_model);

    Filter::StateDistribution state_dist;
    state_dist.initialize(State_a::Zero(dim_state_a, 1),
                          10,
                          State_b::Zero(dim_state_b, 1),
                          1.0,
                          1.0);

    EXPECT_TRUE(state_dist.cov_aa.ldlt().isPositive());
    EXPECT_TRUE(state_dist.joint_partitions[0].cov_bb.ldlt().isPositive());

    for (size_t i = 0; i < 1000; ++i)
    {
        filter.Predict(state_dist, 1.0, state_dist);

        EXPECT_TRUE(state_dist.cov_aa.ldlt().isPositive());
        EXPECT_TRUE(state_dist.joint_partitions[0].cov_bb.ldlt().isPositive());

        Observation y = Observation::Random(dim_observation, 1);
        filter.Update(state_dist, y, state_dist);

        EXPECT_TRUE(state_dist.cov_aa.ldlt().isPositive());
        EXPECT_TRUE(state_dist.joint_partitions[0].cov_bb.ldlt().isPositive());
    }
}

