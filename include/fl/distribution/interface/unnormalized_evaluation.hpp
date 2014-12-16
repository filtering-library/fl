/*
 * This is part of the FL library, a C++ Bayesian filtering library
 * (https://github.com/filtering-library)
 *
 * Copyright (c) 2014 Jan Issac (jan.issac\gmail.com)
 * Copyright (c) 2014 Manuel Wuthrich (manuel.wuthrich\gmail.com)
 *
 * Max-Planck Institute for Intelligent Systems, AMD Lab
 * University of Southern California, CLMC Lab
 *
 * This Source Code Form is subject to the terms of the MIT License (MIT).
 * A copy of the license can be found in the LICENSE file distributed with this
 * source code.
 */

/**
 * \file unnormalized_evaluation.hpp
 * \date May 2014
 * \author Manuel Wuthrich (manuel.wuthrich\gmail.com)
 */


#ifndef FL__DISTRIBUTION__INTERFACE__UNNORMALIZED_EVAUATION_HPP
#define FL__DISTRIBUTION__INTERFACE__UNNORMALIZED_EVAUATION_HPP

#include <cmath>

#include <fl/util/traits.hpp>

namespace fl
{

/**
  \ingroup distribution_interfaces

 * \interface UnnormalizedEvaulation
 */
template <typename Vector, typename Scalar>
class UnnormalizedEvaulation
{
public:
    virtual ~UnnormalizedEvaulation() { }

    /**
     * \brief UnnormalizedProbability bla
     * \param vector
     * \return bla
     */
    virtual Scalar UnnormalizedProbability(const Vector& vector) const
    {
        std::exp(LogUnnormalizedProbability(vector));
    }

    /**
     * \brief LogUnnormalizedProbability bla
     * \param vector
     * \return bla
     */
    virtual Scalar LogUnnormalizedProbability(const Vector& vector) const = 0;
};

}

#endif
