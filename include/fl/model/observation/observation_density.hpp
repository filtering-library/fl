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
 * \file observation_density.hpp
 * \date June 2015
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__MODEL__OBSERVATION__OBSERVATION_DENSITY_HPP
#define FL__MODEL__OBSERVATION__OBSERVATION_DENSITY_HPP

#include <fl/util/traits.hpp>

namespace fl
{

template <
    typename Obsrv,
    typename State
>
class ObservationDensity
{
public:
    /**
     *
     */

    /// \todo should add the unnormalized log probability interface
    virtual FloatingPoint log_probability(const Obsrv& obsrv,
                                          const State& state) const = 0;

    virtual FloatingPoint probability(const Obsrv& obsrv,
                                      const State& state) const
    {
        return std::exp(log_probability(obsrv, state));
    }

    virtual Eigen::Array<FloatingPoint, Eigen::Dynamic, 1> log_probabilities(
        const Obsrv& obsrv,
        const Eigen::Array<State, Eigen::Dynamic, 1>& states)
    {
        Eigen::Array<FloatingPoint, Eigen::Dynamic, 1> probs (states.size());

        for (int i = 0; i < states.size(); ++i)
        {
            probs[i] = log_probability(obsrv, states[i]);
        }

        return probs;
    }

    virtual Eigen::Array<FloatingPoint, Eigen::Dynamic, 1> probabilities(
        const Obsrv& obsrv,
        const Eigen::Array<State, Eigen::Dynamic, 1>& states)
    {
        return log_probabilities(obsrv, states).exp();
    }

    /**
     * \return Dimension of the state variable $\f$x\f$
     */
    virtual int state_dimension() const = 0;

    /**
     * \return Dimension of the measurement \f$h(x, w)\f$
     */
    virtual int obsrv_dimension() const = 0;
};

}

#endif
