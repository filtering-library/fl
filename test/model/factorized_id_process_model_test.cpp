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
 * \file factorized_id_process_model_test.cpp
 * \date January 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <iostream>

#include <fl/model/process/linear_process_model.hpp>
#include <fl/model/process/joint_process_model.hpp>


class FactorizedIDProcessModelTests
        : public ::testing::Test
{
protected:
    typedef fl::LinearGaussianProcessModel<Eigen::Matrix<double, 3, 1>> FModelA;
    typedef fl::LinearGaussianProcessModel<Eigen::Matrix<double, 5, 1>> FModelB;
    typedef fl::LinearGaussianProcessModel<Eigen::Matrix<double, 7, 1>> FModelC;

    typedef fl::LinearGaussianProcessModel<Eigen::Matrix<double, -1, 1>> DModelA;
    typedef fl::LinearGaussianProcessModel<Eigen::Matrix<double, -1, 1>> DModelB;
    typedef fl::LinearGaussianProcessModel<Eigen::Matrix<double, -1, 1>> DModelC;

    typedef typename fl::Traits<FModelA>::SecondMoment FACov;
    typedef typename fl::Traits<FModelB>::SecondMoment FBCov;
    typedef typename fl::Traits<FModelC>::SecondMoment FCCov;
    typedef typename fl::Traits<DModelA>::SecondMoment DACov;
    typedef typename fl::Traits<DModelB>::SecondMoment DBCov;
    typedef typename fl::Traits<DModelC>::SecondMoment DCCov;

    typedef fl::JointProcessModel<FModelA, FModelB, FModelC> FixedModel;
    typedef fl::JointProcessModel<DModelA, DModelB, DModelC> DynamicModel;
    typedef fl::JointProcessModel<FModelA, DModelB, FModelC> DynamicFallbackModel;

};


TEST_F(FactorizedIDProcessModelTests, fixed)
{
    EXPECT_EQ(fl::Traits<FixedModel>::State::SizeAtCompileTime, 15);
    EXPECT_EQ(fl::Traits<FixedModel>::Noise::SizeAtCompileTime, 15);
    EXPECT_EQ(fl::Traits<FixedModel>::Input::SizeAtCompileTime, 3);
}

TEST_F(FactorizedIDProcessModelTests, dynamic)
{
    EXPECT_EQ(fl::Traits<DynamicModel>::State::SizeAtCompileTime, -1);
    EXPECT_EQ(fl::Traits<DynamicModel>::Noise::SizeAtCompileTime, -1);
    EXPECT_EQ(fl::Traits<DynamicModel>::Input::SizeAtCompileTime,  3);
}

TEST_F(FactorizedIDProcessModelTests, dynamic_fallback)
{
    EXPECT_EQ(fl::Traits<DynamicFallbackModel>::State::SizeAtCompileTime, -1);
    EXPECT_EQ(fl::Traits<DynamicFallbackModel>::Noise::SizeAtCompileTime, -1);
    EXPECT_EQ(fl::Traits<DynamicFallbackModel>::Input::SizeAtCompileTime,  3);
}


TEST_F(FactorizedIDProcessModelTests, state_dimension_fixed)
{
    auto my_model = FixedModel(std::make_shared<FModelA>(FACov::Identity()),
                               std::make_shared<FModelB>(FBCov::Identity()),
                               std::make_shared<FModelC>(FCCov::Identity()));

    EXPECT_EQ(my_model.state_dimension(), 15);
}

TEST_F(FactorizedIDProcessModelTests, state_dimension_dynamic)
{
    auto my_model = DynamicModel(
        std::make_shared<DModelA>(DACov::Identity(3,3), 3),
        std::make_shared<DModelB>(DBCov::Identity(5,5), 5),
        std::make_shared<DModelC>(DCCov::Identity(7,7), 7));

    EXPECT_EQ(my_model.state_dimension(), 3 + 5 + 7);
}

TEST_F(FactorizedIDProcessModelTests, state_dimension_dynamic_fallback)
{
    auto my_model = DynamicFallbackModel(
        std::make_shared<FModelA>(FACov::Identity()),
        std::make_shared<DModelB>(DBCov::Identity(5,5), 5),
        std::make_shared<FModelC>(FCCov::Identity()));

    EXPECT_EQ(my_model.state_dimension(), 3 + 5 + 7);
}

TEST_F(FactorizedIDProcessModelTests, predict_fixed)
{
    auto my_model = FixedModel(
        std::make_shared<FModelA>(FACov::Identity() * 3 * 3),
        std::make_shared<FModelB>(FBCov::Identity() * 5 * 5),
        std::make_shared<FModelC>(FCCov::Identity() * 7 * 7));

    auto state = typename fl::Traits<FixedModel>::State();
    auto noise = typename fl::Traits<FixedModel>::Noise();
    auto input = typename fl::Traits<FixedModel>::Input();

    state.setZero();
    noise.setOnes();
    input.setZero();

    EXPECT_EQ(my_model.state_dimension(), 15);
    EXPECT_EQ(my_model.noise_dimension(), 15);
    EXPECT_EQ(my_model.input_dimension(), 3);

    auto prediction = my_model.predict_state(1., state, noise, input);

    auto expected = Eigen::Matrix<double, 15, 1>::Ones().eval();
    expected.topRows(3) = expected.topRows(3) * 3;
    expected.middleRows(3, 5) = expected.middleRows(3, 5) * 5;
    expected.bottomRows(7) = expected.bottomRows(7) * 7;

    EXPECT_TRUE(prediction.isApprox(expected));
}

TEST_F(FactorizedIDProcessModelTests, predict_dynamic)
{
    auto my_model = DynamicModel(
        std::make_shared<DModelA>(DACov::Identity(3, 3) * 3 * 3, 3),
        std::make_shared<DModelB>(DBCov::Identity(5, 5) * 5 * 5, 5),
        std::make_shared<DModelC>(DCCov::Identity(7, 7) * 7 * 7, 7));

    auto state = typename fl::Traits<DynamicModel>::State(15, 1);
    auto noise = typename fl::Traits<DynamicModel>::Noise(15, 1);
    auto input = typename fl::Traits<DynamicModel>::Input(3, 1);

    state.setZero();
    noise.setOnes();
    input.setZero();

    EXPECT_EQ(my_model.state_dimension(), 15);
    EXPECT_EQ(my_model.noise_dimension(), 15);
    EXPECT_EQ(my_model.input_dimension(), 3);

    auto prediction = my_model.predict_state(1., state, noise, input);

    auto expected = Eigen::Matrix<double, 15, 1>::Ones().eval();
    expected.topRows(3) = expected.topRows(3) * 3;
    expected.middleRows(3, 5) = expected.middleRows(3, 5) * 5;
    expected.bottomRows(7) = expected.bottomRows(7) * 7;

    EXPECT_TRUE(prediction.isApprox(expected));
}

TEST_F(FactorizedIDProcessModelTests, predict_dynamic_fallback)
{
    auto my_model = DynamicFallbackModel(
        std::make_shared<FModelA>(FACov::Identity() * 3 * 3),
        std::make_shared<DModelB>(DBCov::Identity(5, 5) * 5 * 5, 5),
        std::make_shared<FModelC>(FCCov::Identity() * 7 * 7));

    auto state = typename fl::Traits<DynamicFallbackModel>::State(15, 1);
    auto noise = typename fl::Traits<DynamicFallbackModel>::Noise(15, 1);
    auto input = typename fl::Traits<DynamicFallbackModel>::Input(3, 1);

    state.setZero();
    noise.setOnes();
    input.setZero();

    EXPECT_EQ(my_model.state_dimension(), 15);
    EXPECT_EQ(my_model.noise_dimension(), 15);
    EXPECT_EQ(my_model.input_dimension(), 3);

    auto prediction = my_model.predict_state(1., state, noise, input);

    auto expected = Eigen::Matrix<double, 15, 1>::Ones().eval();
    expected.topRows(3) = expected.topRows(3) * 3;
    expected.middleRows(3, 5) = expected.middleRows(3, 5) * 5;
    expected.bottomRows(7) = expected.bottomRows(7) * 7;

    EXPECT_TRUE(prediction.isApprox(expected));
}

//long int sum = 0;

//TEST_F(FactorizedIDProcessModelTests, speed_test_fixed)
//{
//    typedef fl::FactorizedIDProcessModel<
//                FModelA, FModelB, FModelC,
//                FModelA, FModelB, FModelC,
//                FModelA, FModelB, FModelC,
//                FModelA, FModelB, FModelC,
//                FModelA, FModelB, FModelC,
//                FModelA, FModelB, FModelC,
//                FModelA, FModelB, FModelC,
//                FModelA, FModelB, FModelC,
//                FModelA, FModelB, FModelC,
//                FModelA, FModelB, FModelC,
//                FModelA, FModelB, FModelC,
//                FModelA, FModelB, FModelC,
//                FModelA, FModelB, FModelC,
//                FModelA, FModelB, FModelC,
//                FModelA, FModelB, FModelC> LargeModel;


//    auto my_model = LargeModel(
//        std::make_shared<FModelA>(FACov::Identity()),
//        std::make_shared<FModelB>(FBCov::Identity()),
//        std::make_shared<FModelC>(FCCov::Identity()),
//        std::make_shared<FModelA>(FACov::Identity()),
//        std::make_shared<FModelB>(FBCov::Identity()),
//        std::make_shared<FModelC>(FCCov::Identity()),
//        std::make_shared<FModelA>(FACov::Identity()),
//        std::make_shared<FModelB>(FBCov::Identity()),
//        std::make_shared<FModelC>(FCCov::Identity()),
//        std::make_shared<FModelA>(FACov::Identity()),
//        std::make_shared<FModelB>(FBCov::Identity()),
//        std::make_shared<FModelC>(FCCov::Identity()),
//        std::make_shared<FModelA>(FACov::Identity()),
//        std::make_shared<FModelB>(FBCov::Identity()),
//        std::make_shared<FModelC>(FCCov::Identity()),
//        std::make_shared<FModelA>(FACov::Identity()),
//        std::make_shared<FModelB>(FBCov::Identity()),
//        std::make_shared<FModelC>(FCCov::Identity()),
//        std::make_shared<FModelA>(FACov::Identity()),
//        std::make_shared<FModelB>(FBCov::Identity()),
//        std::make_shared<FModelC>(FCCov::Identity()),
//        std::make_shared<FModelA>(FACov::Identity()),
//        std::make_shared<FModelB>(FBCov::Identity()),
//        std::make_shared<FModelC>(FCCov::Identity()),
//        std::make_shared<FModelA>(FACov::Identity()),
//        std::make_shared<FModelB>(FBCov::Identity()),
//        std::make_shared<FModelC>(FCCov::Identity()),
//        std::make_shared<FModelA>(FACov::Identity()),
//        std::make_shared<FModelB>(FBCov::Identity()),
//        std::make_shared<FModelC>(FCCov::Identity()),
//        std::make_shared<FModelA>(FACov::Identity()),
//        std::make_shared<FModelB>(FBCov::Identity()),
//        std::make_shared<FModelC>(FCCov::Identity()),
//        std::make_shared<FModelA>(FACov::Identity()),
//        std::make_shared<FModelB>(FBCov::Identity()),
//        std::make_shared<FModelC>(FCCov::Identity()),
//        std::make_shared<FModelA>(FACov::Identity()),
//        std::make_shared<FModelB>(FBCov::Identity()),
//        std::make_shared<FModelC>(FCCov::Identity()),
//        std::make_shared<FModelA>(FACov::Identity()),
//        std::make_shared<FModelB>(FBCov::Identity()),
//        std::make_shared<FModelC>(FCCov::Identity()),
//        std::make_shared<FModelA>(FACov::Identity()),
//        std::make_shared<FModelB>(FBCov::Identity()),
//        std::make_shared<FModelC>(FCCov::Identity()));

//    sum = 0;

//    for (int i = 0; i < 1000000; ++i)
//    {
//        sum += my_model.state_dimension();
//    }
//}


//TEST_F(FactorizedIDProcessModelTests, speed_test_dynamic)
//{
//    auto my_model = DynamicModel(
//        std::make_shared<DModelA>(DACov::Identity(3,3), 3),
//        std::make_shared<DModelB>(DBCov::Identity(5,5), 5),
//        std::make_shared<DModelC>(DCCov::Identity(7,7), 7));

//    sum = 0;

//    for (int i = 0; i < 100000000; ++i)
//    {
//        sum += my_model.state_dimension();
//    }
//}

//TEST_F(FactorizedIDProcessModelTests, speed_test_dynamic_fallback)
//{
//    auto my_model = DynamicFallbackModel(
//        std::make_shared<FModelA>(FACov::Identity()),
//        std::make_shared<DModelB>(DBCov::Identity(5,5), 5),
//        std::make_shared<FModelC>(FCCov::Identity()));

//    sum = 0;

//    for (int i = 0; i < 100000000; ++i)
//    {
//        sum += my_model.state_dimension();
//    }
//}


