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
 * \file joint_distribution_iid_test.cpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <fl/distribution/gaussian.hpp>
#include <fl/distribution/joint_distribution_iid.hpp>

TEST(JointDistribution_IID_Tests, fixed_fixed_single_gaussian)
{
    constexpr int dim = 2;

    typedef Eigen::Matrix<double, dim, 1> FVariate;

    typedef fl::JointDistribution<
                fl::MultipleOf<fl::Gaussian<FVariate>, 1>
            > FSingeDistribution;

    FSingeDistribution joint_distr =
        FSingeDistribution(fl::Gaussian<FVariate>());

    typedef Eigen::Matrix<double, dim, 1> ExpectedMean;
    typedef Eigen::Matrix<double, dim, dim> ExpectedCovariance;

    EXPECT_EQ(joint_distr.dimension(), dim);
    EXPECT_TRUE(joint_distr.mean().isApprox(ExpectedMean::Zero(dim, 1)));
    EXPECT_TRUE(joint_distr
                    .covariance()
                    .isApprox(ExpectedCovariance::Identity(dim, dim)));
}

TEST(JointDistribution_IID_Tests, fixed_dynamic_single_gaussian)
{
    typedef Eigen::Matrix<double, -1, 1> DVariate;

    typedef fl::JointDistribution<
                fl::MultipleOf<fl::Gaussian<DVariate>, 1>
            > FSingeDistribution;

    FSingeDistribution joint_distr =
        FSingeDistribution(fl::Gaussian<DVariate>(13));

    constexpr int dim = 13;

    typedef Eigen::Matrix<double, dim, 1> ExpectedMean;
    typedef Eigen::Matrix<double, dim, dim> ExpectedCovariance;

    EXPECT_EQ(joint_distr.dimension(), dim);
    EXPECT_TRUE(joint_distr.mean().isApprox(ExpectedMean::Zero(dim, 1)));
    EXPECT_TRUE(joint_distr
                    .covariance()
                    .isApprox(ExpectedCovariance::Identity(dim, dim)));
}

TEST(JointDistribution_IID_Tests, dynamic_dynamic_single_gaussian)
{
    typedef Eigen::Matrix<double, -1, 1> DVariate;

    typedef fl::JointDistribution<
                fl::MultipleOf<fl::Gaussian<DVariate>, -1>
            > FSingeDistribution;

    FSingeDistribution joint_distr =
        FSingeDistribution(fl::Gaussian<DVariate>(13), 1);

    constexpr int dim = 13;

    typedef Eigen::Matrix<double, dim, 1> ExpectedMean;
    typedef Eigen::Matrix<double, dim, dim> ExpectedCovariance;

    EXPECT_EQ(joint_distr.dimension(), dim);
    EXPECT_TRUE(joint_distr.mean().isApprox(ExpectedMean::Zero(dim, 1)));
    EXPECT_TRUE(joint_distr
                    .covariance()
                    .isApprox(ExpectedCovariance::Identity(dim, dim)));
}

TEST(JointDistribution_IID_Tests, fixed_fixed_three_gaussian)
{
    typedef Eigen::Matrix<double, 13, 1> FVariate;

    typedef fl::JointDistribution<
                fl::MultipleOf<fl::Gaussian<FVariate>, 3>
            > FSingeDistribution;

    FSingeDistribution joint_distr =
        FSingeDistribution(fl::Gaussian<FVariate>());

    constexpr int dim =  3 * 13;

    typedef Eigen::Matrix<double, dim, 1> ExpectedMean;
    typedef Eigen::Matrix<double, dim, dim> ExpectedCovariance;

    EXPECT_EQ(joint_distr.dimension(), dim);
    EXPECT_TRUE(joint_distr.mean().isApprox(ExpectedMean::Zero(dim, 1)));
    EXPECT_TRUE(joint_distr
                    .covariance()
                    .isApprox(ExpectedCovariance::Identity(dim, dim)));
}

TEST(JointDistribution_IID_Tests, fixed_dynamic_three_gaussian)
{
    typedef Eigen::Matrix<double, -1, 1> DVariate;

    typedef fl::JointDistribution<
                fl::MultipleOf<fl::Gaussian<DVariate>, 3>
            > FSingeDistribution;

    FSingeDistribution joint_distr =
        FSingeDistribution(fl::Gaussian<DVariate>(13));

    constexpr int dim =  3 * 13;

    typedef Eigen::Matrix<double, dim, 1> ExpectedMean;
    typedef Eigen::Matrix<double, dim, dim> ExpectedCovariance;

    EXPECT_EQ(joint_distr.dimension(), dim);
    EXPECT_TRUE(joint_distr.mean().isApprox(ExpectedMean::Zero(dim, 1)));
    EXPECT_TRUE(joint_distr
                    .covariance()
                    .isApprox(ExpectedCovariance::Identity(dim, dim)));
}

TEST(JointDistribution_IID_Tests, dynamic_dynamic_three_gaussian)
{
    typedef Eigen::Matrix<double, -1, 1> DVariate;

    typedef fl::JointDistribution<
                fl::MultipleOf<fl::Gaussian<DVariate>, -1>
            > FSingeDistribution;

    FSingeDistribution joint_distr =
        FSingeDistribution(fl::Gaussian<DVariate>(13), 3);

    constexpr int dim =  3 * 13;

    typedef Eigen::Matrix<double, dim, 1> ExpectedMean;
    typedef Eigen::Matrix<double, dim, dim> ExpectedCovariance;

    EXPECT_EQ(joint_distr.dimension(), dim);
    EXPECT_TRUE(joint_distr.mean().isApprox(ExpectedMean::Zero(dim, 1)));
    EXPECT_TRUE(joint_distr
                    .covariance()
                    .isApprox(ExpectedCovariance::Identity(dim, dim)));
}
