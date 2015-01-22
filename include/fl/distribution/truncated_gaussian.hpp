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
 * \file truncated_gaussian.hpp
 * \date May 2014
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#ifndef FL__DISTRIBUTION__TRUNCATED_GAUSSIAN_HPP
#define FL__DISTRIBUTION__TRUNCATED_GAUSSIAN_HPP


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
class TruncatedGaussian
        : public Evaluation<double, double>,
          public StandardGaussianMapping<double, double>
{
public:
    TruncatedGaussian(double mean = 0.0,
                      double sigma = 1.0,
                      double min = -std::numeric_limits<double>::infinity(),
                      double max = std::numeric_limits<double>::infinity())
        : mean_(mean),
          sigma_(sigma),
          min_(min),
          max_(max)
    {
        ComputeAuxiliaryParameters();
    }

    virtual ~TruncatedGaussian() { }

    virtual void parameters(double mean, double sigma, double min, double max)
    {
        mean_ =  mean;
        sigma_ = sigma;
        min_ =   min;
        max_ =   max;

        ComputeAuxiliaryParameters();
    }

    virtual double probability(const double& input) const
    {
        if(input < min_ || input > max_)
            return 0;

        return normalization_factor_ *
                std::exp(-0.5 * std::pow((input-mean_)/sigma_, 2));
    }

    virtual double log_probability(const double& input) const
    {
        return std::log(probability(input));
    }

    virtual double map_standard_normal(const double& gaussian_sample) const
    {
        // map from a gaussian to a uniform distribution
        double standard_uniform_sample = 0.5 *
                (1.0 + std::erf(gaussian_sample / std::sqrt(2.0)));
        // map onto truncated uniform distribution
        double truncated_uniform_sample = cumulative_min_ +
                  standard_uniform_sample * (cumulative_max_ - cumulative_min_);
        // map onto truncated gaussian

        return mean_ + sigma_ * std::sqrt(2.0) *
                  fl::erfinv(2.0 * truncated_uniform_sample - 1.0);
    }

private:
    virtual void ComputeAuxiliaryParameters()
    {
        cumulative_min_ = 0.5 +
                0.5 * std::erf( (min_-mean_) / (sigma_*std::sqrt(2)) );
        cumulative_max_ = 0.5 +
                0.5 * std::erf( (max_-mean_) / (sigma_*std::sqrt(2)) );

        normalization_factor_ = 1.0 /
             (sigma_ * (cumulative_max_-cumulative_min_) * std::sqrt(2.0*M_PI));
    }

private:
    double mean_;
    double sigma_;
    double min_;
    double max_;

    double cumulative_min_;
    double cumulative_max_;
    double normalization_factor_;
};

}

#endif
