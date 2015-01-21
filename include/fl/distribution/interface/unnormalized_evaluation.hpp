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
 * \file unnormalized_evaluation.hpp
 * \date May 2014
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */


#ifndef FL__DISTRIBUTION__INTERFACE__UNNORMALIZED_EVAUATION_HPP
#define FL__DISTRIBUTION__INTERFACE__UNNORMALIZED_EVAUATION_HPP

#include <cmath>

namespace fl
{

/**
 * \interface UnnormalizedEvaluation
 * \ingroup distribution_interfaces
 *
 * UnnormalizedEvaluation represents the part of a distribution interface
 * which evaluates the unnormalized probability \f$\tilde{p}(x)\f$ of a sample
 * \f$x\f$ according to the implemented distribution. Unnormalized in this
 * context means the integral
 *
 * \f[ \int\limits_{-\infty}^{\infty} \tilde{p}(x) dx \in (-\infty; \infty) \f]
 *
 * does not necessarliy integrate to 1.
 */
template <typename Variate, typename Scalar>
class UnnormalizedEvaluation
{
public:
    virtual ~UnnormalizedEvaluation() { }

    /**
     * Determines the normalized or unnormalized probability for the specified
     * variate.
     *
     * \param variate  The sample \f$x\f$ to evaluate
     *
     * \return \f$\tilde{p}(x)\f$
     */
    virtual Scalar unnormalized_probability(const Variate& variate) const
    {
        std::exp(log_unnormalized_probability(variate));
    }

    /**
     * The log of the unnormalized probability of the specified variate
     *
     * \param variate  The sample \f$x\f$ to evaluate
     *
     * \return \f$\ln(\tilde{p}(x))\f$
     */
    virtual Scalar log_unnormalized_probability(const Variate& variate) const = 0;
};

}

#endif
