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

#pragma once


#include <fl/util/types.hpp>
#include <fl/util/traits.hpp>
#include <fl/distribution/interface/approximate_moments.hpp>

namespace fl
{

#ifndef GENERATING_DOCUMENTATION
// Forward declaration
template <typename...> class Moments;
#endif

/**
 * \ingroup distribution_interfaces
 *
 * \brief Represents the interface providing the first two moments
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
template <typename Variate_, typename SecondMoment_>
#ifndef GENERATING_DOCUMENTATION
class Moments<Variate_, SecondMoment_>
#else
class Moments
#endif
    : public ApproximateMoments<Variate_, SecondMoment_>
{
public:
    /**
     * \brief Variate Random variable type. This is equivalent to the first
     *        moment type.
     */
    typedef Variate_ Variate;

    /**
     * \brief Second central moment type (e.g. Variance or the Covariance)
     */
    typedef SecondMoment_ SecondMoment;

    /**
     * \brief Overridable default destructor
     */
    virtual ~Moments() noexcept { }

    /**
     * \brief Returns the first moment of the underlying distribution, the mean
     *
     * \f$ \mu = \sum\limits_i x_i p(x_i)\f$
     */
    virtual const Variate& mean() const = 0;

    /**
     * \brief Returns the second centered moment of the underlying distribution,
     *         the covariance
     *
     * \f$ \Sigma =
     *     \sum\limits_i (x_i - \mu)(x_i - \mu)^T \f$
     */
    virtual const SecondMoment& covariance() const = 0;

    const Variate& approximate_mean() const override
    {
        return mean();
    }

    const SecondMoment& approximate_covariance() const override
    {
        return covariance();
    }
};

/**
 * \ingroup distribution_interfaces
 *
 */
template <typename Variate>
class Moments<Variate>
    : public Moments<Variate, typename SecondMomentOf<Variate>::Type>
{
public:
    /**
     * \brief Overridable default destructor
     */
    virtual ~Moments() noexcept { }
};

/**
 * \ingroup distribution_interfaces
 *
 */
template < >
class Moments<Real>
    : public Moments<Real, Real>
{
public:
    /**
     * \brief Overridable default destructor
     */
    virtual ~Moments() noexcept { }
};

}
