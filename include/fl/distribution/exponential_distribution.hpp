/*
 * This is part of the fl library, a C++ Bayesian filtering library
 * (https://github.com/filtering-library)
 *
 * Copyright (c) 2015 Max Planck Society,
 * 				 Autonomous Motion Department,
 * 			     Institute for Intelligent Systems
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

#pragma once


#include <limits>
#include <cmath>

#include <fl/util/types.hpp>
#include <fl/distribution/interface/evaluation.hpp>
#include <fl/distribution/interface/standard_gaussian_mapping.hpp>

namespace fl
{

/// \todo MISSING DOC. MISSING UTESTS

/**
 * \ingroup distributions
 */
class ExponentialDistribution:
        public Evaluation<Real>,
        public StandardGaussianMapping<Real, 1>
{

public:
    ExponentialDistribution(Real lambda,
                            Real min = 0,
                            Real max = std::numeric_limits<Real>::infinity()):
                                            lambda_(lambda),
                                            min_(min),
                                            max_(max)
    {
        exp_lambda_min_ = std::exp(-lambda_*min);
        exp_lambda_max_ = std::exp(-lambda_*max);
    }

    virtual ~ExponentialDistribution() noexcept { }

    virtual Real probability(const Real& input) const
    {
        if(input < min_ || input > max_)
            return 0;

        return lambda_ * std::exp(-lambda_ * input) /
                            (exp_lambda_min_ - exp_lambda_max_);
    }

    virtual Real log_probability(const Real& input) const
    {
        return std::log(probability(input));
    }

    virtual Real map_standard_normal(const Real& gaussian_sample) const
    {
        // map from a gaussian to a uniform distribution
        Real uniform_sample = 0.5 *
                        (1.0 + std::erf(gaussian_sample / std::sqrt(2.0)));
        // map from a uniform to an exponential distribution
        return -std::log(exp_lambda_min_ - (exp_lambda_min_ - exp_lambda_max_)
                            * uniform_sample) / lambda_;
    }

    virtual Real map_standard_normal(const Real& gaussian_sample,
                                       const Real& max) const
    {
        Real exp_lambda_max = std::exp(-lambda_*max);

        // map from a gaussian to a uniform distribution
        Real uniform_sample = 0.5 *
                        (1.0 + std::erf(gaussian_sample / std::sqrt(2.0)));
        // map from a uniform to an exponential distribution
        return -std::log(exp_lambda_min_ - (exp_lambda_min_ - exp_lambda_max)
                            * uniform_sample) / lambda_;
    }

private:
    Real lambda_;
    Real min_;
    Real max_;
    Real exp_lambda_min_;
    Real exp_lambda_max_;
};

}
