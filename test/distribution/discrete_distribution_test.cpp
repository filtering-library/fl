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
 * @date 2015
 * @author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * Max-Planck-Institute for Intelligent Systems
 */

#include <gtest/gtest.h>

#include <vector>

#include <fl/distribution/discrete_distribution.hpp>
#include <fl/distribution/gaussian.hpp>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>






TEST(discrete_distribution, moments)
{
    typedef Eigen::Vector3d Variate;
    typedef Eigen::Matrix3d Covariance;
    typedef Variate::Scalar Scalar;
    typedef fl::DiscreteDistribution<Variate> DiscreteDistribution;
    typedef DiscreteDistribution::Function Function;

    // pick some mean and covariance
    Covariance covariance;
    covariance  <<  4.4, 2.1, -1.3,
                    2.2, 5.6,  1.2,
                   -1.2, 1.9,  3.9;
    covariance = covariance * covariance.transpose();

    Variate mean;
    mean << 2.1, 50.2, 20.1;

    // create gaussian
    fl::Gaussian<Variate> gaussian;
    gaussian.mean(mean);
    gaussian.covariance(covariance);

    // generate a sum of delta from gaussian
    DiscreteDistribution discrete_distribution;
    discrete_distribution.log_unnormalized_prob_mass(Function::Zero(100000));

    for(size_t i = 0; i < discrete_distribution.size(); i++)
    {
        discrete_distribution.location(i) = gaussian.sample();
    }

    // compare mean and covariance
    Covariance covariance_delta =
           discrete_distribution.covariance().inverse() * gaussian.covariance();

    EXPECT_TRUE(covariance_delta.isApprox(Covariance::Identity(), 0.1));

    EXPECT_TRUE((gaussian.square_root().inverse() *
                             (discrete_distribution.mean()-mean)).norm() < 0.1);
}


TEST(discrete_distribution, entropy)
{
    typedef Eigen::Vector3d Variate;
    typedef Variate::Scalar Scalar;
    typedef fl::DiscreteDistribution<Variate> DiscreteDistribution;
    typedef DiscreteDistribution::Function Function;

    int N = 100000;

    // check entropy of uniform distribution
    DiscreteDistribution discrete_distribution;
    discrete_distribution.log_unnormalized_prob_mass(Function::Zero(N));

    EXPECT_TRUE(fabs(std::log(double(discrete_distribution.size()))
                     - discrete_distribution.entropy()) < 0.0000001);

    // check entropy of certain distribution
    Function log_pmf = Function::Constant(N,-std::numeric_limits<double>::max());
    log_pmf(0) = 0;
    discrete_distribution.log_unnormalized_prob_mass(log_pmf);


    EXPECT_TRUE(fabs(discrete_distribution.entropy()) < 0.0000001);
}


TEST(discrete_distribution, sampling)
{
    typedef Eigen::Matrix<int, 1, 1> Variate;
    typedef fl::DiscreteDistribution<Variate> DiscreteDistribution;
    typedef DiscreteDistribution::Function Function;
    typedef std::vector<Variate> Locations;

    int N_locations = 10;
    int N_samples   = 1000000;

    // random prob mass fct
    Function pmf = Function::Random(N_locations).abs() + 0.01;
    pmf /= pmf.sum();

    // create discrete distr
    DiscreteDistribution discrete_distribution;
    discrete_distribution.log_unnormalized_prob_mass(pmf.log());

    for(int i = 0; i < N_locations; i++)
        discrete_distribution.location(i)(0) = i;

    // generate empirical pmf
    Function empirical_pmf = Function::Zero(N_locations);
    for(int i = 0; i < N_samples; i++)
    {
        empirical_pmf(discrete_distribution.sample()(0)) += 1./N_samples;
    }

    // make sure that pmf and empirical pmf are similar
    for(int i = 0; i < pmf.size(); i++)
    {
        EXPECT_TRUE(fabs(pmf[i] - empirical_pmf[i]) < 0.01);
    }
}










