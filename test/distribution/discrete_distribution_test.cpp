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

    EXPECT_TRUE(fabs(discrete_distribution.kl_given_uniform()) < 0.0000001);

    // check entropy of certain distribution
    Function log_pmf = Function::Constant(N,-std::numeric_limits<double>::max());
    log_pmf(0) = 0;
    discrete_distribution.log_unnormalized_prob_mass(log_pmf);

    EXPECT_TRUE(fabs(discrete_distribution.entropy()) < 0.0000001);

    EXPECT_TRUE(fabs(std::log(double(discrete_distribution.size()))
                     - discrete_distribution.kl_given_uniform()) < 0.0000001);
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
