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

#ifndef FL__DISTRIBUTION__INTERFACE__MOMENTS_HPP
#define FL__DISTRIBUTION__INTERFACE__MOMENTS_HPP

#include <fl/distribution/interface/approximate_moments.hpp>

namespace fl
{

/**
 * \interface Moments
 * \ingroup distribution_interfaces
 *
 * \brief Moments is the interface providing the first two moments
 *
 * \tparam Variate        Random variable type. This is equivalent to the first
 *                        moment type.
 * \tparam SecondMoment   Second moment type. The second moment is either
 *                        the second uncentered moment \f$Var(X) + X^2\f$ or
 *                        simply the second central moment, the variance or
 *                        covariance \f$Var(X) = Cov(X, X)\f$. Both have the
 *                        same type \c SecondMoment.
 *
 *
 * The Moments interface provides access to the exact first moments of
 * a distribution. The moments represent a subset of the approximate moments.
 */
template <typename Variate, typename SecondMoment>
class Moments:
    public ApproximateMoments<Variate, SecondMoment>
{
public:    
    /**
     * \brief Overridable default destructor
     */
    virtual ~Moments() { }

    /**
     * \return First moment of the underlying distribution, the mean
     *
     * \f$ \mu = \sum\limits_i x_i p(x_i)\f$
     */
    virtual Variate mean() const = 0;

    /**
     * \return Second centered moment of the underlying distribution,
     *         the covariance
     *
     * \f$ \Sigma =
     *     \sum\limits_i (x_i - \mu)(x_i - \mu)^T \f$
     */
    virtual SecondMoment covariance() const = 0;

    /**
     * \copydoc ApproximateMoments::approximate_mean
     */
    virtual Variate approximate_mean() const
    {
        return mean();
    }

    /**
     * \copydoc ApproximateMoments::approximate_covariance
     */
    virtual SecondMoment approximate_covariance() const
    {
        return covariance();
    }
};

}

#endif
