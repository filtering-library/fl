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

#pragma once


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
    : public Evaluation<Vector1d>,
      public StandardGaussianMapping<Vector1d, 1>
{
private:
    typedef StandardGaussianMapping<Vector1d, 1> StdGaussianMappingBase;

public:
    typedef Vector1d Variate;

    /**
     * \brief Represents the StandardGaussianMapping standard variate type which
     *        is of the same dimension as the \c TDistribution \c Variate. The
     *        StandardVariate type is used to sample from a standard normal
     *        Gaussian and map it to this \c TDistribution
     */
    typedef typename StdGaussianMappingBase::StandardVariate StandardVariate;

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

    virtual ~UniformDistribution() noexcept { }

    Real probability(const Variate& x) const override
    {
        assert(x.size() == 1);

        if(x(0) < min_ || x(0) > max_) return 0;

        return density_;
    }

    Real log_probability(const Variate& x) const override
    {
        assert(x.size() == 1);

        if(x(0) < min_ || x(0) > max_)
        {
            return -std::numeric_limits<Real>::infinity();
        }

        return log_density_;
    }

    Variate map_standard_normal(const StandardVariate& sample) const override
    {
        assert(sample.size() == 1);

        // map from a gaussian to a uniform distribution
        Real standard_uniform_sample = fl::normal_to_uniform(sample(0));

        Variate v;
        v(0) = mean_ + (standard_uniform_sample - 0.5) * delta_;

        return v;
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
