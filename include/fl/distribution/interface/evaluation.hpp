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

#include <fl/util/types.hpp>
#include <fl/distribution/interface/unnormalized_evaluation.hpp>

namespace fl
{

/**
 * \ingroup distribution_interfaces
 *
 * \brief Distribution evaulation interface
 *
 * \tparam Variate  Random variable type
 * \tparam Scalar   Probability & log scalar type
 *
 * Evaluation provides the interface to determine the probability of the
 * underlying distribution at a given sample. Evaluation is a subset of
 * unnormalized distributions. Normalized in this context means
 * \f[ \int\limits_{-\infty}^{\infty} p(x) dx = 1. \f]
 */
template <typename Variate>
class Evaluation:
        public UnnormalizedEvaluation<Variate>
{
public:
    /**
     * \brief Overridable default destructor
     */
    virtual ~Evaluation() { }

    /**
     * Determines the probability for the specified variate.
     *
     * \param variate Sample \f$x\f$ to evaluate
     *
     * \return \f$p(x)\f$
     */
    virtual Real probability(const Variate& variate) const
    {
        return std::exp(log_probability(variate));
    }

    /**
     * Determines the probability log for the specified variate.
     *
     * \param variate Sample \f$x\f$ to evaluate
     *
     * \return \f$\ln(p(x))\f$
     */
    virtual Real log_probability(const Variate& variate) const = 0;

    /**
     * \copydoc UnnormalizedEvaluation::log_unnormalized_probability
     */
    virtual Real log_unnormalized_probability(const Variate& variate) const
    {
        return log_probability(variate);
    }
};

}

#endif
