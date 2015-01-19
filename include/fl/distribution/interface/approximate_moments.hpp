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

#ifndef FL__DISTRIBUTION__INTERFACE__APPROXIMATE_CENTRAL_MOMENTS_HPP
#define FL__DISTRIBUTION__INTERFACE__APPROXIMATE_CENTRAL_MOMENTS_HPP

namespace fl
{

/**
 * \interface ApproximateCentralMoments
 * \ingroup distribution_interfaces
 *
 * \brief ApproximateMoments is the interface providing the first two moments
 *
 * \tparam Variate          Random variable type. This is equivalent to the
 *                          first moment type.
 * \tparam SecondMoment     Second moment type (e.g. Variance or the Covariance)
 *
 * The ApproximateMoments interface provides access to a numerical approximation
 * of the first moments of a distribution.
 *
 */
template <typename Variate, typename SecondMoment>
class ApproximateCentralMoments
{
public:
    /**
     * \brief Overridable default destructor
     */
    virtual ~ApproximateCentralMoments() { }

    /**
     * \return First moment approximation, the mean
     */
    virtual Variate approximate_mean() const = 0;

    /**
     * \return Second centered moment, the covariance
     */
    virtual SecondMoment approximate_covariance() const = 0;
};

}

#endif
