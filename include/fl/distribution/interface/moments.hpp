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
 * \brief Moments is the interface providing the first two moments
 *
 * \tparam Vector   Random variable type. This is equivalent to the first moment
 *                  type.
 * \tparam Operator Second moment type
 *
 * The Moments interface provides access to the exact first moments of
 * a distribution. The moments represent a subset of the approximate moments.
 */
template <typename Vector, typename Operator>
class Moments:
    public ApproximateMoments<Vector, Operator>
{
public:    

    /**
     * \return First moment of the underlying distribution, the mean
     */
    virtual Vector Mean() const = 0;

    /**
     * \return Second centered moment of the underlying distribution,
     *         the covariance
     */
    virtual Operator Covariance() const = 0;

    /**
     * \{ApproximateMoments::ApproximateMean() const\}
     */
    virtual Vector ApproximateMean() const
    {
        return Mean();
    }

    /**
     * \{ApproximateMoments::ApproximateCovariance() const\}
     */
    virtual Operator ApproximateCovariance() const
    {
        return Covariance();
    }

    /**
     * \brief Overridable default destructor
     */
    virtual ~Moments() { }
};

}

#endif
