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

#include <fl/distribution/sum_of_deltas.hpp>
#include <fl/distribution/gaussian.hpp>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>




TEST(sum_of_delta, mean_and_covariance)
{
    typedef Eigen::Vector3d Variate;
    typedef Eigen::Matrix3d Covariance;
    typedef Variate::Scalar Scalar;

    typedef fl::SumOfDeltas<Variate> DiscreteDistribution;
    typedef DiscreteDistribution::Probabilities Function;

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
    DiscreteDistribution sum_of_delta;
    sum_of_delta.log_unnormalized_probabilities(Function::Zero(100000));

    for(size_t i = 0; i < sum_of_delta.size(); i++)
    {
        sum_of_delta.location(i) = gaussian.sample();
    }

    // compare mean and covariance
    Covariance covariance_delta =
            sum_of_delta.covariance().inverse() * gaussian.covariance();

    EXPECT_TRUE(covariance_delta.isApprox(Covariance::Identity(), 0.1));

    EXPECT_TRUE((gaussian.square_root().inverse() *
                                    (sum_of_delta.mean()-mean)).norm() < 0.1);

    // check entropy of uniform distribution
    sum_of_delta.log_unnormalized_probabilities(Function::Zero(sum_of_delta.size()));

    EXPECT_TRUE(fabs(std::log(double(sum_of_delta.size()))
                     - sum_of_delta.entropy()) < 0.0000001);

    // check entropy of certain distribution
    Function log_pmf = Function::Constant(sum_of_delta.size(),
                                          -std::numeric_limits<double>::max());
    log_pmf(0) = 0;
    sum_of_delta.log_unnormalized_probabilities(log_pmf);


    EXPECT_TRUE(fabs(sum_of_delta.entropy()) < 0.0000001);
}

