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
 * \file exponential_distribution.hpp
 * \date May 2014
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#ifndef FAST_FILTERING_DISTRIBUTIONS_EXPONENTIAL_DISTRIBUTION_HPP
#define FAST_FILTERING_DISTRIBUTIONS_EXPONENTIAL_DISTRIBUTION_HPP


#include <limits>
#include <cmath>


#include <fl/distribution/interface/evaluation.hpp>
#include <fl/distribution/interface/gaussian_map.hpp>

namespace fl
{

class ExponentialDistribution:
        public Evaluation<double, double>,
        public GaussianMap<double, double>
{

public:
    ExponentialDistribution(double lambda,
                            double min = 0,
                            double max = std::numeric_limits<double>::infinity()):
                                            lambda_(lambda),
                                            min_(min),
                                            max_(max)
    {
        exp_lambda_min_ = std::exp(-lambda_*min);
        exp_lambda_max_ = std::exp(-lambda_*max);
    }


    virtual ~ExponentialDistribution() { }

    virtual double Probability(const double& input) const
    {
        if(input < min_ || input > max_)
            return 0;

        return lambda_ * std::exp(-lambda_ * input) /
                            (exp_lambda_min_ - exp_lambda_max_);
    }

    virtual double LogProbability(const double& input) const
    {
        return std::log(Probability(input));
    }

    virtual double MapStandardGaussian(const double& gaussian_sample) const
    {
        // map from a gaussian to a uniform distribution
        double uniform_sample = 0.5 *
                        (1.0 + std::erf(gaussian_sample / std::sqrt(2.0)));
        // map from a uniform to an exponential distribution
        return -std::log(exp_lambda_min_ - (exp_lambda_min_ - exp_lambda_max_)
                            * uniform_sample) / lambda_;
    }


    virtual double MapStandardGaussian(const double& gaussian_sample,
                                       const double& max) const
    {
        double exp_lambda_max = std::exp(-lambda_*max);

        // map from a gaussian to a uniform distribution
        double uniform_sample = 0.5 *
                        (1.0 + std::erf(gaussian_sample / std::sqrt(2.0)));
        // map from a uniform to an exponential distribution
        return -std::log(exp_lambda_min_ - (exp_lambda_min_ - exp_lambda_max)
                            * uniform_sample) / lambda_;
    }

private:
    double lambda_;
    double min_;
    double max_;
    double exp_lambda_min_;
    double exp_lambda_max_;
};

}

#endif
