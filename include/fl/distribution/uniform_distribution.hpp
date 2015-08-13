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
 * \file uniform_distribution.hpp
 * \date May 2014
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#ifndef FL__DISTRIBUTION__UNIFORM_DISTRIBUTION_HPP
#define FL__DISTRIBUTION__UNIFORM_DISTRIBUTION_HPP

#include <limits>
#include <cmath>

#include <fl/util/math.hpp>
#include <fl/distribution/interface/evaluation.hpp>
#include <fl/distribution/interface/standard_gaussian_mapping.hpp>

namespace fl
{

/// \todo MISSING DOC. MISSING UTESTS

/**
 * \ingroup distributions
 */
class UniformDistribution
    : public Evaluation<Real>,
      public StandardGaussianMapping<Real, 1>
{

public:
    UniformDistribution()
        : min_(0.), max_(1.)
    {
        init();
    }

    UniformDistribution(Real min, Real max)
        : min_(min), max_(max)
    {
        init();
    }

    virtual ~UniformDistribution() { }

    Real probability(const Real& input) const override
    {
        if(input < min_ || input > max_) return 0;

        return density_;
    }

    Real log_probability(const Real& input) const override
    {
        if(input < min_ || input > max_)
        {
            return -std::numeric_limits<Real>::infinity();
        }

        return log_density_;
    }

    Real map_standard_normal(const Real& gaussian_sample) const override
    {
        // map from a gaussian to a uniform distribution
        Real standard_uniform_sample = fl::normal_to_uniform(gaussian_sample);

        return mean_ + (standard_uniform_sample - 0.5) * delta_;
    }

private:
    void init()
    {
        delta_ = max_ - min_;
        density_ = 1.0 / delta_;
        mean_ = (max_ + min_) / 2.0;
        log_density_ = std::log(density_);
    }

    Real min_;
    Real max_;
    Real delta_;
    Real mean_;
    Real density_;
    Real log_density_;
};

}

#endif
