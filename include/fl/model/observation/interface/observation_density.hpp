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

#include <fl/util/types.hpp>
#include <fl/util/traits.hpp>

namespace fl
{

template <
    typename Obsrv,
    typename State,
    int BatchSize = Eigen::Dynamic
>
class ObservationDensity
{
public:
    typedef Eigen::Array<State, BatchSize, 1> StateArray;
    typedef Eigen::Array<Real,  BatchSize, 1> ValueArray;

public:
    /**
     * \brief Overridable default destructor
     */
    virtual ~ObservationDensity() { }

    /// \todo should add the unnormalized log probability interface

    virtual Real log_probability(const Obsrv& obsrv,
                                 const State& state) const = 0;

    /**
     * \return Dimension of the state variable $\f$x\f$
     */
    virtual int state_dimension() const = 0;

    /**
     * \return Dimension of the measurement \f$h(x, w)\f$
     */
    virtual int obsrv_dimension() const = 0;

    virtual Real probability(const Obsrv& obsrv,
                             const State& state) const
    {
        return std::exp(log_probability(obsrv, state));
    }

    virtual ValueArray log_probabilities(const Obsrv& obsrv,
                                         const StateArray& states)
    {
        auto probs = ValueArray(states.size());

        for (int i = 0; i < states.size(); ++i)
        {
            probs[i] = log_probability(obsrv, states[i]);
        }

        return probs;
    }

    virtual ValueArray probabilities(const Obsrv& obsrv,
                                     const StateArray& states)
    {
        return log_probabilities(obsrv, states).exp();
    }
};




template <
    typename Obsrv,
    typename State,
    int BatchSize = Eigen::Dynamic
>
class SwitchingObservationDensity
{
public:
    typedef Eigen::Array<State, BatchSize, 1>  StateArray;
    typedef Eigen::Array<Real, BatchSize, 1>   ValueArray;
    typedef Eigen::Array<int, BatchSize, 1>    IndexArray;

public:
    /**
     * \brief Overridable default destructor
     */
    virtual ~SwitchingObservationDensity() { }

    /// \todo should add the unnormalized log probability interface

    virtual Real log_probability(const Obsrv& obsrv,
                                          const State& state,
                                          const int&   index) const = 0;

    /**
     * \return Dimension of the state variable $\f$x\f$
     */
    virtual int state_dimension() const = 0;

    /**
     * \return Dimension of the measurement \f$h(x, w)\f$
     */
    virtual int obsrv_dimension() const = 0;


    virtual Real probability(const Obsrv& obsrv,
                             const State& state,
                             const int&   index) const
    {
        return std::exp(log_probability(obsrv, state, index));
    }

    virtual ValueArray log_probabilities(const Obsrv& obsrv,
                                         const StateArray& states,
                                         const IndexArray& indices)
    {
        auto probs = ValueArray(states.size());

        for (int i = 0; i < states.size(); ++i)
        {
            probs[i] = log_probability(obsrv, states[i], indices[i]);
        }

        return probs;
    }

    virtual ValueArray probabilities(const Obsrv& obsrv,
                                     const StateArray& states,
                                     const IndexArray& indices)
    {
        return log_probabilities(obsrv, states, indices).exp();
    }
};

}

#endif
