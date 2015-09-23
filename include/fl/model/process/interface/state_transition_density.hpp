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
 * \file process_density.hpp
 * \date June 2015
 * \author Jan Issac (jan.issac@gmail.com)
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#pragma once


#include <fl/util/types.hpp>
#include <fl/util/traits.hpp>

namespace fl
{

template <
    typename State_,
    typename Input_,
    int BatchSize = Eigen::Dynamic
>
class StateTransitionDensity
{
public:
    typedef State_ State;
    typedef Input_ Input;

    typedef Eigen::Array<State, BatchSize, 1 > StateArray;
    typedef Eigen::Array<Input, BatchSize, 1 > InputArray;
    typedef Eigen::Array<Real, BatchSize, 1 > ValueArray;

public:
    /**
     * \brief Overridable default destructor
     */
    virtual ~StateTransitionDensity() noexcept { }


    /// \todo should add the unnormalized log probability interface


    virtual Real log_probability(const State& state,
                                 const State& cond_state,
                                 const Input& cond_input) const = 0;

    /**
     * \return Dimension of the state variable $\fx_t\f$
     */
    virtual int state_dimension() const = 0;

    /**
     * \return Dimension of the input \f$u_t\f$
     */
    virtual int input_dimension() const = 0;


    virtual Real probability(const State& state,
                             const State& cond_state,
                             const Input& cond_input) const
    {
        return std::exp(log_probability(state,
                                        cond_state,
                                        cond_input));
    }

    virtual ValueArray log_probabilities(const StateArray& states,
                                         const StateArray& cond_states,
                                         const InputArray& cond_inputs) const
    {
        assert(states.size() == cond_inputs.size());

        auto probs = ValueArray(states.size());

        for (int i = 0; i < states.size(); ++i)
        {
            probs[i] = log_probability(states[i],
                                       cond_states[i],
                                       cond_inputs[i]);
        }

        return probs;
    }

    virtual ValueArray probabilities(const StateArray& states,
                                     const StateArray& cond_states,
                                     const InputArray& cond_inputs)
    {
        return log_probabilities(states, cond_states, cond_inputs).exp();
    }
};

}


