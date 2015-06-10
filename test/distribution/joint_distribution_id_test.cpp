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
 * \file joint_distribution_id_test.cpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <iostream>

#include <fl/distribution/gaussian.hpp>
#include <fl/distribution/joint_distribution_id.hpp>

TEST(JointDistribution_ID_Tests, fixed_accessing_marginals)
{
    typedef Eigen::Matrix<double, 2, 1> FVariate;
    typedef fl::JointDistribution<fl::Gaussian<FVariate>> FSingeDistribution;

    FSingeDistribution joint_distr =
        FSingeDistribution(fl::Gaussian<FVariate>());

    typedef fl::Traits<
                std::tuple_element<
                    0,
                    fl::Traits<FSingeDistribution>::MarginalDistributions
                >::type
            >::Variate ExpectedMean;

    typedef fl::Traits<
                std::tuple_element<
                    0,
                    fl::Traits<FSingeDistribution>::MarginalDistributions
                >::type
            >::SecondMoment ExpectedCovariance;

    EXPECT_EQ(joint_distr.dimension(), 2);
    EXPECT_TRUE(fl::are_similar(joint_distr.mean(), ExpectedMean::Zero()));
    EXPECT_TRUE(fl::are_similar(joint_distr.covariance(),
                                ExpectedCovariance::Identity()));

    EXPECT_TRUE(fl::are_similar(
                    std::get<0>(joint_distr.distributions()).mean(),
                    ExpectedMean::Zero()));

    EXPECT_TRUE(fl::are_similar(
                    std::get<0>(joint_distr.distributions()).covariance(),
                    ExpectedCovariance::Identity()));
}


TEST(JointDistribution_ID_Tests, fixed_single_gaussian)
{
    typedef Eigen::Matrix<double, 2, 1> FVariate;
    typedef fl::JointDistribution<fl::Gaussian<FVariate>> FSingeDistribution;

    FSingeDistribution joint_distr =
        FSingeDistribution(fl::Gaussian<FVariate>());

    constexpr int dim = 2;

    typedef Eigen::Matrix<double, dim, 1> ExpectedMean;
    typedef Eigen::Matrix<double, dim, dim> ExpectedCovariance;

    EXPECT_EQ(joint_distr.dimension(), 2);
    EXPECT_TRUE(fl::are_similar(joint_distr.mean(), ExpectedMean::Zero()));
    EXPECT_TRUE(fl::are_similar(joint_distr.covariance(),
                                ExpectedCovariance::Identity()));

    EXPECT_TRUE(fl::are_similar(
                    std::get<0>(joint_distr.distributions()).mean(),
                    ExpectedMean::Zero()));

    EXPECT_TRUE(fl::are_similar(
                    std::get<0>(joint_distr.distributions()).covariance(),
                    ExpectedCovariance::Identity()));
}

TEST(JointDistribution_ID_Tests, fixed_two_gaussians)
{
    typedef Eigen::Matrix<double, 2, 1> FVariate2D;
    typedef Eigen::Matrix<double, 7, 1> FVariate7D;
    typedef fl::JointDistribution<
                fl::Gaussian<FVariate2D>,
                fl::Gaussian<FVariate7D>> FJointDistribution;

    FJointDistribution joint_distr =
        FJointDistribution(
            fl::Gaussian<FVariate2D>(),
            fl::Gaussian<FVariate7D>());

    constexpr int dim = 2 + 7;

    typedef Eigen::Matrix<double, dim, 1> ExpectedMean;
    typedef Eigen::Matrix<double, dim, dim> ExpectedCovariance;

    EXPECT_EQ(joint_distr.dimension(), dim);
    EXPECT_TRUE(fl::are_similar(joint_distr.mean(), ExpectedMean::Zero()));
    EXPECT_TRUE(fl::are_similar(joint_distr.covariance(),
                                ExpectedCovariance::Identity()));
}

TEST(JointDistribution_ID_Tests, fixed_three_gaussians)
{
    typedef Eigen::Matrix<double, 2, 1> FVariate2D;
    typedef Eigen::Matrix<double, 7, 1> FVariate7D;
    typedef Eigen::Matrix<double, 13, 1> FVariate13D;

    typedef fl::JointDistribution<
                fl::Gaussian<FVariate2D>,
                fl::Gaussian<FVariate13D>,
                fl::Gaussian<FVariate7D>> FJointDistribution;

    FJointDistribution joint_distr =
        FJointDistribution(
            fl::Gaussian<FVariate2D>(),
            fl::Gaussian<FVariate13D>(),
            fl::Gaussian<FVariate7D>());

    constexpr int dim = 2 + 13 + 7;

    typedef Eigen::Matrix<double, dim, 1> ExpectedMean;
    typedef Eigen::Matrix<double, dim, dim> ExpectedCovariance;

    EXPECT_EQ(joint_distr.dimension(), dim);
    EXPECT_TRUE(fl::are_similar(joint_distr.mean(), ExpectedMean::Zero()));
    EXPECT_TRUE(fl::are_similar(joint_distr.covariance(),
                                ExpectedCovariance::Identity()));
}

TEST(JointDistribution_ID_Tests, fixed_recursive_definition)
{
    typedef Eigen::Matrix<double, 2, 1> FVariate2D;
    typedef Eigen::Matrix<double, 7, 1> FVariate7D;

    typedef fl::JointDistribution<
                fl::Gaussian<FVariate7D>
            > InnerJointDistribution;

    typedef fl::JointDistribution<
                fl::Gaussian<FVariate2D>,
                InnerJointDistribution
            > FJointDistribution;

    FJointDistribution joint_distr =
        FJointDistribution(
            fl::Gaussian<FVariate2D>(),
            InnerJointDistribution(fl::Gaussian<FVariate7D>()));

    constexpr int dim = 2 + 7;

    typedef Eigen::Matrix<double, dim, 1> ExpectedMean;
    typedef Eigen::Matrix<double, dim, dim> ExpectedCovariance;

    EXPECT_EQ(joint_distr.dimension(), dim);
    EXPECT_TRUE(fl::are_similar(joint_distr.mean(), ExpectedMean::Zero()));
    EXPECT_TRUE(fl::are_similar(joint_distr.covariance(),
                                ExpectedCovariance::Identity()));
}

TEST(JointDistribution_ID_Tests, fixed_deep_recursive_definition)
{
    typedef Eigen::Matrix<double, 2, 1> FVariate2D;
    typedef Eigen::Matrix<double, 7, 1> FVariate7D;

    typedef fl::JointDistribution<
                fl::Gaussian<FVariate7D>
            > InnerInnerJointDistribution;

    typedef fl::JointDistribution<
                InnerInnerJointDistribution,
                InnerInnerJointDistribution
            > InnerJointDistribution;

    typedef fl::JointDistribution<
                fl::Gaussian<FVariate2D>,
                InnerJointDistribution
            > FJointDistribution;

    FJointDistribution joint_distr =
        FJointDistribution(
            fl::Gaussian<FVariate2D>(),
            InnerJointDistribution(
                InnerInnerJointDistribution(fl::Gaussian<FVariate7D>()),
                InnerInnerJointDistribution(fl::Gaussian<FVariate7D>())));

    constexpr int dim = 2 + 2 * 7;

    typedef Eigen::Matrix<double, dim, 1> ExpectedMean;
    typedef Eigen::Matrix<double, dim, dim> ExpectedCovariance;

    EXPECT_EQ(joint_distr.dimension(), dim);
    EXPECT_TRUE(fl::are_similar(joint_distr.mean(), ExpectedMean::Zero()));
    EXPECT_TRUE(fl::are_similar(joint_distr.covariance(),
                                ExpectedCovariance::Identity()));
}

TEST(JointDistribution_ID_Tests, dynamic_single_gaussian)
{
    constexpr int dim = 2;

    typedef Eigen::Matrix<double, -1, 1> DVariate;
    typedef fl::JointDistribution<fl::Gaussian<DVariate>> DSingeDistribution;

    DSingeDistribution joint_distr =
        DSingeDistribution(fl::Gaussian<DVariate>(dim));

    typedef Eigen::Matrix<double, dim, 1> ExpectedMean;
    typedef Eigen::Matrix<double, dim, dim> ExpectedCovariance;

    EXPECT_EQ(joint_distr.dimension(), 2);

    EXPECT_TRUE(fl::are_similar(joint_distr.mean(), ExpectedMean::Zero(dim, 1)));
    EXPECT_TRUE(fl::are_similar(joint_distr.covariance(),
                                ExpectedCovariance::Identity(dim, dim)));

    EXPECT_TRUE(fl::are_similar(
                    std::get<0>(joint_distr.distributions()).mean(),
                    ExpectedMean::Zero(dim, 1)));

    EXPECT_TRUE(fl::are_similar(
                    std::get<0>(joint_distr.distributions()).covariance(),
                    ExpectedCovariance::Identity(dim , dim)));
}

TEST(JointDistribution_ID_Tests, dynamic_three_gaussians)
{
    typedef Eigen::Matrix<double, -1, 1> DVariate;

    constexpr int dim2 = 2;
    constexpr int dim13 = 13;
    constexpr int dim7 =  7;

    constexpr int dim =  dim2 + dim13 + dim7;

    typedef fl::JointDistribution<
                fl::Gaussian<DVariate>,
                fl::Gaussian<DVariate>,
                fl::Gaussian<DVariate>> FJointDistribution;

    FJointDistribution joint_distr =
        FJointDistribution(
            fl::Gaussian<DVariate>(dim2),
            fl::Gaussian<DVariate>(dim13),
            fl::Gaussian<DVariate>(dim7));

    typedef Eigen::Matrix<double, dim, 1> ExpectedMean;
    typedef Eigen::Matrix<double, dim, dim> ExpectedCovariance;

    EXPECT_EQ(joint_distr.dimension(), dim);
    EXPECT_TRUE(fl::are_similar(joint_distr.mean(), ExpectedMean::Zero(dim, 1)));
    EXPECT_TRUE(fl::are_similar(joint_distr.covariance(),
                                ExpectedCovariance::Identity(dim, dim)));
}
