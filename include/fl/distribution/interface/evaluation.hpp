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
 * \file evaluation.hpp
 * \date May 2014
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * \author Jan Issac (jan.issac@gmail.com)
 */


#ifndef FL__DISTRIBUTION__INTERFACE__EVAUATION_HPP
#define FL__DISTRIBUTION__INTERFACE__EVAUATION_HPP

#include <cmath>

#include <fl/distribution/interface/unnormalized_evaluation.hpp>

namespace fl
{

/**
 * \interface Evaluation
 * \brief Distribution evaulation interface
 *
 * \tparam Vector   Random variable type
 * \tparam Scalar   Probability & log scalar type
 *
 * Evaluation provides the interface to determine the probability of the
 * underlying distribution at a given sample. Evaluation is a subset of
 * unnormalized distributions.
 */
template <typename Vector, typename Scalar>
class Evaluation:
        public UnnormalizedEvaulation<Vector, Scalar>
{
public:
    /**
     * \return Normalized probability of a given sample
     *
     * \param vector Sample to evaluate
     */
    virtual Scalar Probability(const Vector& vector) const
    {
        return std::exp(LogProbability(vector));
    }

    /**
     * \return Log of normalized probability of a given sample
     */
    virtual Scalar LogProbability(const Vector& vector) const = 0;

    /**
     * \{UnnormalizedEvaulation
     *   ::LogUnnormalizedProbability(const Vector&) const\}
     */
    virtual Scalar LogUnnormalizedProbability(const Vector& vector) const
    {
        return LogProbability(vector);
    }

    /**
     * \brief Overridable default destructor
     */
    virtual ~Evaluation() { }
};

}

#endif
