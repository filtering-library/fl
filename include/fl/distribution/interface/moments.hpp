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
 * \file moments.hpp
 * \date May 2014
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__DISTRIBUTION__INTERFACE__CENTRAL_MOMENTS_HPP
#define FL__DISTRIBUTION__INTERFACE__CENTRAL_MOMENTS_HPP

#include <fl/distribution/interface/approximate_moments.hpp>

namespace fl
{

/**
 * \interface CentralMoments
 * \ingroup distribution_interfaces
 *
 * \brief CentralMoments is the interface providing the first two moments
 *
 * \tparam Variate        Random variable type. This is equivalent to the first
 *                        moment type.
 * \tparam SecondMoment   Second moment type (e.g. Variance or the Covariance)
 *
 * The CentralMoments interface provides access to the exact first moments of
 * a distribution. The moments represent a subset of the approximate moments.
 */
template <typename Variate, typename SecondMoment>
class CentralMoments:
    public ApproximateCentralMoments<Variate, SecondMoment>
{
public:    
    /**
     * \brief Overridable default destructor
     */
    virtual ~CentralMoments() { }

    /**
     * \return First moment of the underlying distribution, the mean
     */
    virtual Variate mean() const = 0;

    /**
     * \return Second centered moment of the underlying distribution,
     *         the covariance
     */
    virtual SecondMoment covariance() const = 0;

    /**
     * \{ApproximateCentralMoments::Approximatemean() const\}
     */
    virtual Variate approximate_mean() const
    {
        return mean();
    }

    /**
     * \{ApproximateCentralMoments::Approximatecovariance() const\}
     */
    virtual SecondMoment approximate_covariance() const
    {
        return covariance();
    }
};

}

#endif
