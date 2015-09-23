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
 * \file approximate_moments.hpp
 * \date May 2014
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once


#include <fl/util/types.hpp>
#include <fl/util/traits.hpp>

namespace fl
{

// Forward declaration
template <typename...> class ApproximateMoments;

/**
 * \ingroup distribution_interfaces
 *
 * \brief Represents the interface providing the first two central moments
 *
 * \tparam Variate          Random variable type. This is equivalent to the
 *                          first moment type.
 *
 * The ApproximateMoments interface provides access to a numerical approximation
 * of the first moments of a distribution.
 *
 */
template <typename Variate_, typename SecondMoment_>
class ApproximateMoments<Variate_, SecondMoment_>
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
    virtual ~ApproximateMoments() noexcept { }

    /**
     * \brief Returns the first moment approximation, the mean
     *
     * \f$ \mu_{approx} \approx \sum\limits_i x_i p(x_i)\f$
     */
    virtual const Variate& approximate_mean() const = 0;

    /**
     * \brief Returns the second centeral moment, the covariance
     *
     * \f$ \Sigma_{approx} \approx
     *     \sum\limits_i (x_i - \mu)(x_i - \mu)^T \f$
     */
    virtual const SecondMoment& approximate_covariance() const = 0;
};

/**
 * \ingroup distribution_interfaces
 */
template <typename Variate>
class ApproximateMoments<Variate>
    : public ApproximateMoments<Variate, typename SecondMomentOf<Variate>::Type>
{
public:
    /**
     * \brief Overridable default destructor
     */
    virtual ~ApproximateMoments() noexcept { }
};

/**
 * \ingroup distribution_interfaces
 */
template <>
class ApproximateMoments<Real>
    : public ApproximateMoments<Real, Real>
{
public:
    /**
     * \brief Overridable default destructor
     */
    virtual ~ApproximateMoments() noexcept { }
};

}
