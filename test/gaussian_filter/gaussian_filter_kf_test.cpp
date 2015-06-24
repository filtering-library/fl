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
 * \file gaussian_filter_kf_test.cpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>
#include "../typecast.hpp"

#include <Eigen/Dense>

#include "gaussian_filter_test_suite.hpp"

template <typename TestType>
class KalmanFilterTest
    : public GaussianFilterTest<TestType>
{
public:
    typedef GaussianFilterTest<TestType> Base;

    typedef typename Base::LinearStateTransition LinearStateTransition;
    typedef typename Base::LinearObservation LinearObservation;

    typedef fl::GaussianFilter<LinearStateTransition, LinearObservation> Filter;

    static Filter create_kalman_filter()
    {
        auto filter = Filter(
                        LinearStateTransition(Base::StateDim, Base::InputDim),
                        LinearObservation(Base::ObsrvDim, Base::StateDim));

        return filter;
    }
};

typedef ::testing::Types<
            fl::StaticTest,
            fl::DynamicTest
        > TestTypes;

TYPED_TEST_CASE(KalmanFilterTest, TestTypes);

TYPED_TEST(KalmanFilterTest, init_predict)
{
    auto filter = TestFixture::create_kalman_filter();
    auto belief = filter.create_belief();

    predict(filter, belief);
}

TYPED_TEST(KalmanFilterTest, predict_then_update)
{
    auto filter = TestFixture::create_kalman_filter();
    auto belief = filter.create_belief();

    predict_update(filter, belief);
}

TYPED_TEST(KalmanFilterTest, predict_and_update)
{
    auto filter = TestFixture::create_kalman_filter();

    auto belief_A  = filter.create_belief();
    auto belief_B  = filter.create_belief();

    predict_and_update(filter, belief_A, belief_B);
}

//TEST(KalmanFilterTests, fixed_size_predict_loop)
//{
//    constexpr int StateDim = 6;
//    constexpr int ObsrvDim = 6;
//    constexpr int InputDim = 6;


//    typedef Eigen::Matrix<Real, StateDim, 1> State;
//    typedef Eigen::Matrix<Real, ObsrvDim, 1> Input;
//    typedef Eigen::Matrix<Real, InputDim, 1> Obsrv;

//    typedef LinearStateTransitionModel<State, Input> LinearStateTransition;
//    typedef LinearObservationModel<Obsrv, State> LinearObservation;

//    // the KalmanFilter
//    typedef GaussianFilter<LinearStateTransition, LinearObservation> Filter;

//    auto filter = Filter(LinearStateTransition(StateDim, InputDim),
//                         LinearObservation(ObsrvDim, StateDim));

//    auto belief  = Filter::Belief(StateDim);

//    auto A = filter.process_model().create_dynamics_matrix();
//    auto Q = filter.process_model().create_noise_matrix();

//    A.setIdentity();
//    Q.setIdentity();

//    filter.process_model().dynamics_matrix(A);
//    filter.process_model().noise_matrix(Q);

//    EXPECT_TRUE(belief.covariance().ldlt().isPositive());
//    for (int i = 0; i < 100000; ++i)
//    {
//        filter.predict(belief, Input(InputDim), belief);
//    }
//    EXPECT_TRUE(belief.covariance().ldlt().isPositive());
//}

//TEST(KalmanFilterTests, fixed_size_predict_multiple_loop)
//{
//    constexpr int StateDim = 6;
//    constexpr int ObsrvDim = 6;
//    constexpr int InputDim = 6;


//    typedef Eigen::Matrix<Real, StateDim, 1> State;
//    typedef Eigen::Matrix<Real, ObsrvDim, 1> Input;
//    typedef Eigen::Matrix<Real, InputDim, 1> Obsrv;

//    typedef LinearStateTransitionModel<State, Input> LinearStateTransition;
//    typedef LinearObservationModel<Obsrv, State> LinearObservation;

//    // the KalmanFilter
//    typedef GaussianFilter<LinearStateTransition, LinearObservation> Filter;

//    auto filter = Filter(LinearStateTransition(StateDim, InputDim),
//                         LinearObservation(ObsrvDim, StateDim));

//    auto belief  = Filter::Belief(StateDim);

//    auto A = filter.process_model().create_dynamics_matrix();
//    auto Q = filter.process_model().create_noise_matrix();

//    A.setIdentity();
//    Q.setIdentity();

//    filter.process_model().dynamics_matrix(A);
//    filter.process_model().noise_matrix(Q);

//    EXPECT_TRUE(belief.covariance().ldlt().isPositive());
//    for (int i = 0; i < 100000; ++i)
//    {
//        filter.predict(belief, Input(InputDim), 1, belief);
//    }
//    EXPECT_TRUE(belief.covariance().ldlt().isPositive());
//}

//TEST(KalmanFilterTests, fixed_size_predict_multiple)
//{
//    constexpr int StateDim = 6;
//    constexpr int ObsrvDim = 6;
//    constexpr int InputDim = 6;


//    typedef Eigen::Matrix<Real, StateDim, 1> State;
//    typedef Eigen::Matrix<Real, ObsrvDim, 1> Input;
//    typedef Eigen::Matrix<Real, InputDim, 1> Obsrv;

//    typedef LinearStateTransitionModel<State, Input> LinearStateTransition;
//    typedef LinearObservationModel<Obsrv, State> LinearObservation;

//    // the KalmanFilter
//    typedef GaussianFilter<LinearStateTransition, LinearObservation> Filter;

//    auto filter = Filter(LinearStateTransition(StateDim, InputDim),
//                         LinearObservation(ObsrvDim, StateDim));

//    auto belief  = Filter::Belief(StateDim);

//    auto A = filter.process_model().create_dynamics_matrix();
//    auto Q = filter.process_model().create_noise_matrix();

//    A.setIdentity();
//    Q.setIdentity();

//    filter.process_model().dynamics_matrix(A);
//    filter.process_model().noise_matrix(Q);

//    EXPECT_TRUE(belief.covariance().ldlt().isPositive());
//    filter.predict(belief, Input(InputDim), 100000, belief);
//    EXPECT_TRUE(belief.covariance().ldlt().isPositive());
//}

//TEST(KalmanFilterTests, fixed_size_predict_loop_vs_multiple)
//{
//    constexpr int StateDim = 6;
//    constexpr int ObsrvDim = 6;
//    constexpr int InputDim = 6;


//    typedef Eigen::Matrix<Real, StateDim, 1> State;
//    typedef Eigen::Matrix<Real, ObsrvDim, 1> Input;
//    typedef Eigen::Matrix<Real, InputDim, 1> Obsrv;

//    typedef LinearStateTransitionModel<State, Input> LinearStateTransition;
//    typedef LinearObservationModel<Obsrv, State> LinearObservation;

//    // the KalmanFilter
//    typedef GaussianFilter<LinearStateTransition, LinearObservation> Filter;

//    auto filter = Filter(LinearStateTransition(StateDim, InputDim),
//                         LinearObservation(ObsrvDim, StateDim));

//    auto belief_A  = Filter::Belief(StateDim);
//    auto belief_B = Filter::Belief(StateDim);

//    auto A = filter.process_model().create_dynamics_matrix();
//    auto Q = filter.process_model().create_noise_matrix();

//    A.setIdentity();
//    Q.setIdentity();

//    filter.process_model().dynamics_matrix(A);
//    filter.process_model().noise_matrix(Q);

//    EXPECT_TRUE(belief_A.covariance().ldlt().isPositive());
//    for (int i = 0; i < 2000; ++i)
//    {
//        filter.predict(belief_A, Input(InputDim), belief_A);
//    }
//    filter.predict(belief_B, Input(InputDim), 2000, belief_B);

//    EXPECT_TRUE(belief_A.covariance().ldlt().isPositive());
//    EXPECT_TRUE(belief_B.covariance().ldlt().isPositive());

//    EXPECT_TRUE(
//        fl::are_similar(belief_A.mean(), belief_B.mean()));
//    EXPECT_TRUE(
//        fl::are_similar(belief_A.covariance(), belief_B.covariance()));
//}
