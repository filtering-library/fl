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

#include <Eigen/Dense>

#include <cmath>
#include <iostream>
#include <vector>
#include <ctime>
#include <memory>

#include <boost/iterator/zip_iterator.hpp>
#include <boost/range.hpp>
#include <boost/make_shared.hpp>

#include <fast_filtering/utils/traits.hpp>
#include <fast_filtering/filters/deterministic/kalman_filter.hpp>
#include <fast_filtering/models/process_models/linear_process_model.hpp>
#include <fast_filtering/models/observation_models/linear_observation_model.hpp>
#include <fast_filtering/filters/deterministic/composed_state_distribution.hpp>
#include <fast_filtering/filters/deterministic/factorized_unscented_kalman_filter.hpp>

#define DIM_STATE_A             11
#define DIM_STATE_B             5
#define COUNT_STATE_B           7
#define DIM_OBSERVATION         3
#define DIM_JOINT_STATE         (DIM_STATE_A + DIM_STATE_B * COUNT_STATE_B)
#define DIM_JOINT_OBSERVATION   (DIM_OBSERVATION * COUNT_STATE_B)

#define COEFF_A 0.5465454
#define COEFF_Q 0.06
#define COEFF_R 0.05

#define VERBOSE false

//#define COEFF_A 1.0
//#define COEFF_Q 1.0
//#define COEFF_R 1.0

#define EPSILON 1.e-6

template <typename V1, typename V2> bool areSimilar(V1 v1, V2 v2)
{
    return (v1-v2).dot(v1-v2) < EPSILON;
}

template <typename D1, typename D2> void step_prints(D1& kf_state_dist,
                                                    D2& fukf_state_dist,
                                                    std::string step,
                                                     size_t i)
{
    if (!VERBOSE) return;

    std::cout << "== " << step << " " << i << " =================" << std::endl;

    std::cout << "  kf " << step << " expected "
              << kf_state_dist.Mean().transpose() << std::endl;

    std::cout << "fukf " << step << " expected "
              << fukf_state_dist.mean_a.transpose() ;
    for (auto& partition: fukf_state_dist.joint_partitions)
    {
        std::cout << " " << partition.mean_b.transpose();
    }
    std::cout << std::endl;

    std::cout << "  kf " << step << " cov_aa\n"
                         <<  kf_state_dist
                             .Covariance()
                             .block(0, 0, DIM_STATE_A, DIM_STATE_A)
                         << std::endl;
    std::cout << "fukf " << step << " cov_aa\n"
                         << fukf_state_dist.cov_aa
                         << std::endl;
    std::cout << "  kf " << step << " cov_bb[0] "
                         <<  kf_state_dist.Covariance().block(DIM_STATE_A,
                                                              DIM_STATE_A,
                                                              DIM_STATE_B,
                                                              DIM_STATE_B)
                         << std::endl;
    std::cout << "fukf " << step << " cov_bb[0] "
                         << fukf_state_dist.joint_partitions[0].cov_bb
                         << std::endl;
}

template <typename D1, typename D2> void step_assert(D1& kf_state_dist,
                                                     D2& fukf_state_dist,
                                                     std::string step,
                                                     bool& success)
{
    success = false;

    ASSERT_TRUE(kf_state_dist.Covariance().ldlt().isPositive());
    ASSERT_TRUE(fukf_state_dist.cov_aa.ldlt().isPositive());
    ASSERT_TRUE(fukf_state_dist
                .joint_partitions[0].cov_bb.ldlt().isPositive());

//    ASSERT_TRUE(kf_state_dist
//                .Mean()
//                .topRows(DIM_STATE_A)
//                .isApprox(fukf_state_dist.mean_a));
//    ASSERT_TRUE(kf_state_dist
//                .Mean()
//                .middleRows(DIM_STATE_A, DIM_STATE_B)
//                .isApprox(fukf_state_dist.joint_partitions[0].mean_b));

    ASSERT_TRUE(areSimilar(kf_state_dist.Mean().topRows(DIM_STATE_A),
                           fukf_state_dist.mean_a));
    ASSERT_TRUE(areSimilar(kf_state_dist.Mean().middleRows(DIM_STATE_A,
                                                           DIM_STATE_B),
                           fukf_state_dist.joint_partitions[0].mean_b));

    ASSERT_TRUE(kf_state_dist
                .Covariance()
                .block(0, 0, DIM_STATE_A, DIM_STATE_A)
                .isApprox(fukf_state_dist.cov_aa, EPSILON));
    ASSERT_TRUE(kf_state_dist
                .Covariance()
                .block(DIM_STATE_A, DIM_STATE_A, DIM_STATE_B, DIM_STATE_B)
                .isApprox(fukf_state_dist.joint_partitions[0].cov_bb, EPSILON));

    success = true;
}

TEST(FukfKFTest, init)
{    
    // == COMMON STUFF == //
    typedef double Scalar;

    // == Factorized UKF STUFF == //
    typedef Eigen::Matrix<Scalar, DIM_STATE_A, 1> State_a;
    typedef Eigen::Matrix<Scalar, DIM_STATE_B, 1> State_b;
    typedef Eigen::Matrix<Scalar, DIM_OBSERVATION, 1> Observation_ab;

    typedef ff::LinearGaussianProcessModel<State_a> ProcessModel_a;
    typedef ff::LinearGaussianProcessModel<State_b, State_a> ProcessModel_b;
    typedef ff::FactorizedLinearGaussianObservationModel<
            Observation_ab,
            State_a,
            State_b> FactorizedObservationModel;

    typedef ff::FactorizedUnscentedKalmanFilter<
            ProcessModel_a,
            ProcessModel_b,
            FactorizedObservationModel> FukfFilter;

    // sensor & process dynamics
    FactorizedObservationModel::SensorMatrix_a H_a;
    FactorizedObservationModel::SensorMatrix_b H_b;
    ProcessModel_a::DynamicsMatrix A_a;
    ProcessModel_b::DynamicsMatrix A_b;
    ProcessModel_a::Operator Q_a;
    ProcessModel_b::Operator Q_b;
    FactorizedObservationModel::Operator R_ab;

    H_a.setRandom();
    H_b.setRandom();

    A_a.setIdentity();
    A_b.setIdentity();
    Q_a.setIdentity();
    Q_b.setIdentity();
    R_ab.setIdentity();

    A_a *= COEFF_A;
    A_b *= COEFF_A;
    Q_a *= COEFF_Q;
    Q_b *= COEFF_Q;
    R_ab *= COEFF_R;

    // models
    FukfFilter::CohesiveStateProcessModelPtr process_model_a;
    FukfFilter::FactorizedStateProcessModelPtr process_model_b;
    FukfFilter::ObservationModelPtr observation_model_ab;
    process_model_a = boost::make_shared<ProcessModel_a>(Q_a);
    process_model_b = boost::make_shared<ProcessModel_b>(Q_b);
    observation_model_ab = boost::make_shared<FactorizedObservationModel>(R_ab);

    process_model_a->A(A_a);
    process_model_b->A(A_b);
    observation_model_ab->H_a(H_a);
    observation_model_ab->H_b(H_b);

    FukfFilter fukf_filter(process_model_a,
                           process_model_b,
                           observation_model_ab);

    FukfFilter::StateDistribution fukf_state_dist;
    fukf_state_dist.initialize(State_a::Ones(), COUNT_STATE_B, State_b::Ones());

    // == KalmanFilter STUFF == //
    typedef Eigen::Matrix< Scalar, DIM_JOINT_STATE, 1> State;
    typedef Eigen::Matrix<Scalar, DIM_JOINT_OBSERVATION, 1> Observation;

    typedef ff::LinearGaussianProcessModel<State> ProcessModel;
    typedef ff::LinearGaussianObservationModel<
            Observation,
            State> ObservationModel;

    typedef ff::KalmanFilter<ProcessModel, ObservationModel> KalmanFilter;

    // joint process dynamics
    ProcessModel::DynamicsMatrix A;
    ObservationModel::SensorMatrix H;
    ProcessModel::Operator Q;
    ObservationModel::Operator R;

    // compose model dynamics from the factorized configuration
    H.setZero();
    for (size_t i = 0; i < COUNT_STATE_B; ++i)
    {
        H.block(i*H_a.rows(), 0, H_a.rows(), H_a.cols()) = H_a;

        H.block(i*H_a.rows(), H_a.cols() + i*H_b.cols(),
                H_b.rows(), H_b.cols()) = H_b;
    }

    A.setIdentity();
    Q.setIdentity();
    R.setIdentity();

    A *= COEFF_A;
    Q *= COEFF_Q;
    R *= COEFF_R;

    // joint models
    KalmanFilter::ProcessModelPtr process_model;
    KalmanFilter::ObservationModelPtr observation_model;
    process_model = boost::make_shared<ProcessModel>(Q);
    observation_model = boost::make_shared<ObservationModel>(R);

    process_model->A(A);
    observation_model->H(H);

    KalmanFilter kf_filter(process_model, observation_model);
    KalmanFilter::StateDistribution kf_state_dist;
    kf_state_dist.Mean(State::Ones());

    // assert pre-conditions
    ASSERT_EQ(R.rows(), R.cols());
    ASSERT_EQ(R.rows(), R_ab.rows() * COUNT_STATE_B);
    ASSERT_TRUE(kf_state_dist.Covariance().ldlt().isPositive());

    bool success = false;


    for (size_t i = 0; i < 33; ++i)
    {
        Observation y = Observation::Random();

        // == predict ======================================================= //
        kf_filter.Predict(1.0, kf_state_dist, kf_state_dist);
        fukf_filter.Predict(fukf_state_dist, 1.0, fukf_state_dist);

        step_prints(kf_state_dist, fukf_state_dist, "predict", i);
        step_assert(kf_state_dist, fukf_state_dist, "predict", success);
        if (!success) return;

        // == update ======================================================== //
        kf_filter.Update(kf_state_dist, y, kf_state_dist);
        fukf_filter.Update(fukf_state_dist, y, fukf_state_dist);

        step_prints(kf_state_dist, fukf_state_dist, "update", i);
        step_assert(kf_state_dist, fukf_state_dist, "update", success);
        if (!success) return;

        // == approximate =================================================== //
        ProcessModel::Operator kf_cov = kf_state_dist.Covariance();
        for (size_t i = 0; i < COUNT_STATE_B; ++i)
        {
            kf_cov.block(0, DIM_STATE_A + i*DIM_STATE_B,
                         DIM_STATE_A+ i*DIM_STATE_B, DIM_STATE_B).setZero();

            kf_cov.transpose()
                  .block(0, DIM_STATE_A + i*DIM_STATE_B,
                         DIM_STATE_A+ i*DIM_STATE_B, DIM_STATE_B).setZero();
        }
        kf_state_dist.Covariance(kf_cov);

        ASSERT_TRUE(kf_state_dist.Covariance().ldlt().isPositive()
                    && fukf_state_dist.cov_aa.ldlt().isPositive()
                    && " not p.s.d anymore");
    }
}




























