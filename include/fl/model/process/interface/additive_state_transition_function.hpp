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
 * \file additive_observation_function.hpp
 * \date June 2015
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__MODEL__PROCESS__ADDITIVE_STATE_TRANSITION_FUNCTION_HPP
#define FL__MODEL__PROCESS__ADDITIVE_STATE_TRANSITION_FUNCTION_HPP

#include <fl/util/traits.hpp>

#include <fl/model/process/interface/process_model_interface.hpp>

namespace fl
{

template <
    typename State,
    typename Noise,
    typename Input,
    int Id = 0
>
class AdditiveStateTransitionFunction
    : public StateTransitionFunction<State, Noise, Input, Id>
{
public:
    typedef StateTransitionFunction<State, Noise, Input, Id> FunctionInterface;

    /**
     * Noise model matrix \f$N_t\f$
     */
    typedef Eigen::Matrix<
                typename Noise::Scalar,
                SizeOf<Noise>::Value,
                SizeOf<State>::Value
            > NoiseMatrix;

    virtual State expected_state(const State& prev_state,
                                 const Input& input) const = 0;

    virtual const NoiseMatrix& noise_matrix() const = 0;
    virtual const NoiseMatrix& noise_covariance() const = 0;

    virtual State state(const State& prev_state,
                        const Noise& noise,
                        const Input& input) const
    {
        return expected_state(prev_state, input) + noise_matrix() * noise;
    }
};

}

#endif
