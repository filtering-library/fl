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
 * \file joint_process_mode_iid_test.cpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <memory>

#include <fl/model/process/linear_process_model.hpp>
#include <fl/model/process/joint_process_model.hpp>

/* ############################ */
/* # Static dimensions tests  # */
/* ############################ */

TEST(JointProcessModel_IID_Tests, fixed_fixed_static_dimensions)
{
    using namespace fl;

    typedef LinearGaussianProcessModel<Eigen::Matrix<double, 3, 1>> LocalModel;
    typedef typename Traits<LocalModel>::SecondMoment Cov;

    typedef JointProcessModel<MultipleOf<LocalModel, 3>> JointModel;

    EXPECT_EQ(fl::Traits<JointModel>::State::SizeAtCompileTime, 9);
    EXPECT_EQ(fl::Traits<JointModel>::Noise::SizeAtCompileTime, 9);
    EXPECT_EQ(fl::Traits<JointModel>::Input::SizeAtCompileTime, 3);
}

TEST(JointProcessModel_IID_Tests, fixed_dynamic_static_dimensions)
{
    using namespace fl;

    typedef LinearGaussianProcessModel<Eigen::Matrix<double, -1, 1>> LocalModel;
    typedef typename Traits<LocalModel>::SecondMoment Cov;

    typedef JointProcessModel<MultipleOf<LocalModel, 3>> JointModel;

    EXPECT_EQ(fl::Traits<JointModel>::State::SizeAtCompileTime, -1);
    EXPECT_EQ(fl::Traits<JointModel>::Noise::SizeAtCompileTime, -1);
    EXPECT_EQ(fl::Traits<JointModel>::Input::SizeAtCompileTime, 3);
}

TEST(JointProcessModel_IID_Tests, dynamic_fixed_static_dimensions)
{
    using namespace fl;

    typedef LinearGaussianProcessModel<Eigen::Matrix<double, 3, 1>> LocalModel;
    typedef typename Traits<LocalModel>::SecondMoment Cov;

    typedef JointProcessModel<MultipleOf<LocalModel, -1>> JointModel;

    EXPECT_EQ(fl::Traits<JointModel>::State::SizeAtCompileTime, -1);
    EXPECT_EQ(fl::Traits<JointModel>::Noise::SizeAtCompileTime, -1);
    EXPECT_EQ(fl::Traits<JointModel>::Input::SizeAtCompileTime, -1);

    auto f = JointModel(LocalModel(3), 3);
    EXPECT_EQ(f.state_dimension(), 9);
    EXPECT_EQ(f.noise_dimension(), 9);
    EXPECT_EQ(f.input_dimension(), 3);

    auto f_mof = JointModel(
                MultipleOf<LocalModel, -1>(LocalModel(Cov::Identity()), 3));

    EXPECT_EQ(f_mof.state_dimension(), 9);
    EXPECT_EQ(f_mof.noise_dimension(), 9);
    EXPECT_EQ(f_mof.input_dimension(), 3);
}

TEST(JointProcessModel_IID_Tests, dynamic_dynamic_static_dimensions)
{
    using namespace fl;

    typedef LinearGaussianProcessModel<Eigen::Matrix<double, -1, 1>> LocalModel;
    typedef typename Traits<LocalModel>::SecondMoment Cov;

    typedef JointProcessModel<MultipleOf<LocalModel, -1>> JointModel;

    EXPECT_EQ(fl::Traits<JointModel>::State::SizeAtCompileTime, -1);
    EXPECT_EQ(fl::Traits<JointModel>::Noise::SizeAtCompileTime, -1);
    EXPECT_EQ(fl::Traits<JointModel>::Input::SizeAtCompileTime, -1);
}


/* ############################ */
/* # Runtime dimensions tests # */
/* ############################ */

TEST(JointProcessModel_IID_Tests, fixed_fixed_dimensions)
{
    using namespace fl;

    typedef LinearGaussianProcessModel<Eigen::Matrix<double, 3, 1>> LocalModel;
    typedef typename Traits<LocalModel>::SecondMoment Cov;

    typedef JointProcessModel<MultipleOf<LocalModel, 3>> JointModel;

    auto f = JointModel(LocalModel());
    EXPECT_EQ(f.state_dimension(), 9);
    EXPECT_EQ(f.noise_dimension(), 9);
    EXPECT_EQ(f.input_dimension(), 3);

    auto f_mof = JointModel(MultipleOf<LocalModel, 3>(LocalModel()));
    EXPECT_EQ(f_mof.state_dimension(), 9);
    EXPECT_EQ(f_mof.noise_dimension(), 9);
    EXPECT_EQ(f_mof.input_dimension(), 3);
}

TEST(JointProcessModel_IID_Tests, fixed_dynamic_dimensions)
{
    using namespace fl;

    typedef LinearGaussianProcessModel<Eigen::Matrix<double, -1, 1>> LocalModel;
    typedef typename Traits<LocalModel>::SecondMoment Cov;

    typedef JointProcessModel<MultipleOf<LocalModel, 3>> JointModel;

    auto f = JointModel(LocalModel(3));
    EXPECT_EQ(f.state_dimension(), 9);
    EXPECT_EQ(f.noise_dimension(), 9);
    EXPECT_EQ(f.input_dimension(), 3);

    auto f_mof = JointModel(
        MultipleOf<LocalModel, 3>(LocalModel(Cov::Identity(3,3), 3)));
    EXPECT_EQ(f_mof.state_dimension(), 9);
    EXPECT_EQ(f_mof.noise_dimension(), 9);
    EXPECT_EQ(f_mof.input_dimension(), 3);
}

TEST(JointProcessModel_IID_Tests, dynamic_dynamic_dimensions)
{
    using namespace fl;

    typedef LinearGaussianProcessModel<Eigen::Matrix<double, -1, 1>> LocalModel;
    typedef typename Traits<LocalModel>::SecondMoment Cov;

    typedef JointProcessModel<MultipleOf<LocalModel, -1>> JointModel;

    auto f = JointModel(LocalModel(Cov::Identity(3, 3), 3), 3);

    EXPECT_EQ(f.state_dimension(), 9);
    EXPECT_EQ(f.noise_dimension(), 9);
    EXPECT_EQ(f.input_dimension(), 3);

    auto f_mof = JointModel(
        MultipleOf<LocalModel, -1>(LocalModel(Cov::Identity(3,3), 3), 3));
    EXPECT_EQ(f_mof.state_dimension(), 9);
    EXPECT_EQ(f_mof.noise_dimension(), 9);
    EXPECT_EQ(f_mof.input_dimension(), 3);
}

/* ############################ */
/* # Prediction tests         # */
/* ############################ */

TEST(JointProcessModel_IID_Tests, fixed_fixed_prediction)
{
    using namespace fl;

    typedef LinearGaussianProcessModel<Eigen::Matrix<double, 5, 1>> LocalModel;
    typedef typename Traits<LocalModel>::SecondMoment Cov;

    typedef JointProcessModel<MultipleOf<LocalModel, 3>> JointModel;

    auto f = JointModel(LocalModel(Cov::Identity() * 9.0 ));

    auto state = Traits<JointModel>::State();
    auto noise = Traits<JointModel>::Noise();
    auto input = Traits<JointModel>::Input();

    state.setZero();
    noise.setOnes();
    input.setZero();

    auto prediction = f.predict_state(1., state, noise, input);

    auto expected = Eigen::Matrix<double, 15, 1>::Ones().eval();
    expected.topRows(5) = expected.topRows(5) * 3.;
    expected.middleRows(5, 5) = expected.middleRows(5, 5) * 3.;
    expected.bottomRows(5) = expected.bottomRows(5) * 3.;

    EXPECT_TRUE(prediction.isApprox(expected));
}


TEST(JointProcessModel_IID_Tests, fixed_dynamic_prediction)
{
    using namespace fl;

    typedef LinearGaussianProcessModel<Eigen::Matrix<double, -1, 1>> LocalModel;
    typedef typename Traits<LocalModel>::SecondMoment Cov;

    typedef JointProcessModel<MultipleOf<LocalModel, 3>> JointModel;

    auto f = JointModel(LocalModel(Cov::Identity(5, 5) * 9.0, 5));

    auto state = Traits<JointModel>::State(f.state_dimension(), 1);
    auto noise = Traits<JointModel>::Noise(f.noise_dimension(), 1);
    auto input = Traits<JointModel>::Input(f.input_dimension(), 1);

    state.setZero();
    noise.setOnes();
    input.setZero();

    auto prediction = f.predict_state(1., state, noise, input);

    auto expected = Eigen::Matrix<double, 15, 1>::Ones().eval();
    expected.topRows(5) = expected.topRows(5) * 3.;
    expected.middleRows(5, 5) = expected.middleRows(5, 5) * 3.;
    expected.bottomRows(5) = expected.bottomRows(5) * 3.0;

    EXPECT_TRUE(prediction.isApprox(expected));
}


TEST(JointProcessModel_IID_Tests, dynamic_fixed_prediction)
{
    using namespace fl;

    typedef LinearGaussianProcessModel<Eigen::Matrix<double, 5, 1>> LocalModel;
    typedef typename Traits<LocalModel>::SecondMoment Cov;

    typedef JointProcessModel<MultipleOf<LocalModel, -1>> JointModel;

    auto f = JointModel(LocalModel(Cov::Identity() * 9.0), 3);

    auto state = Traits<JointModel>::State(f.state_dimension(), 1);
    auto noise = Traits<JointModel>::Noise(f.noise_dimension(), 1);
    auto input = Traits<JointModel>::Input(f.input_dimension(), 1);

    state.setZero();
    noise.setOnes();
    input.setZero();

    auto prediction = f.predict_state(1., state, noise, input);

    auto expected = Eigen::Matrix<double, 15, 1>::Ones().eval();
    expected.topRows(5) = expected.topRows(5) * 3.;
    expected.middleRows(5, 5) = expected.middleRows(5, 5) * 3.;
    expected.bottomRows(5) = expected.bottomRows(5) * 3.0;

    EXPECT_TRUE(prediction.isApprox(expected));
}

TEST(JointProcessModel_IID_Tests, dynamic_dynamic_prediction)
{
    using namespace fl;

    typedef LinearGaussianProcessModel<Eigen::Matrix<double, -1, 1>> LocalModel;
    typedef typename Traits<LocalModel>::SecondMoment Cov;

    typedef JointProcessModel<MultipleOf<LocalModel, -1>> JointModel;

    auto f = JointModel(LocalModel(Cov::Identity(5,5) * 9.0, 5), 3);

    auto state = Traits<JointModel>::State(f.state_dimension(), 1);
    auto noise = Traits<JointModel>::Noise(f.noise_dimension(), 1);
    auto input = Traits<JointModel>::Input(f.input_dimension(), 1);

    state.setZero();
    noise.setOnes();
    input.setZero();

    auto prediction = f.predict_state(1., state, noise, input);

    auto expected = Eigen::Matrix<double, 15, 1>::Ones().eval();
    expected.topRows(5) = expected.topRows(5) * 3.;
    expected.middleRows(5, 5) = expected.middleRows(5, 5) * 3.;
    expected.bottomRows(5) = expected.bottomRows(5) * 3.0;

    EXPECT_TRUE(prediction.isApprox(expected));
}
