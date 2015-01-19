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

#include <fl/distribution/interface/evaluation.hpp>
#include <fl/distribution/interface/gaussian_map.hpp>

namespace fl
{

/// \todo MISSING DOC. MISSING UTESTS

class UniformDistribution
        : public Evaluation<double, double>,
          public GaussianMap<double, double>
{

public:
    UniformDistribution(double min = 0.0,
                        double max = 1.0): min_(min), max_(max)
    {
        delta_ = max_ - min_;
        density_ = 1.0 / delta_;
        mean_ = (max_ + min_) / 2.0;
        log_density_ = std::log(density_);
    }

    virtual ~UniformDistribution() { }

    virtual double probability(const double& input) const
    {
        if(input < min_ || input > max_)
            return 0;

        return density_;
    }

    virtual double log_probability(const double& input) const
    {
        if(input < min_ || input > max_)
            return -std::numeric_limits<double>::infinity();

        return log_density_;
    }

    virtual double map_standard_normal(const double& gaussian_sample) const
    {
        // map from a gaussian to a uniform distribution
        double standard_uniform_sample =
                0.5 * (1.0 + std::erf(gaussian_sample / std::sqrt(2.0)));
        return mean_ + (standard_uniform_sample - 0.5) * delta_;
    }

private:
    double min_;
    double max_;
    double delta_;
    double mean_;
    double density_;
    double log_density_;
};

}

#endif
