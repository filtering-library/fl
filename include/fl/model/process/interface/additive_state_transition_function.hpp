/*
 * This is part of the fl library, a C++ Bayesian filtering library
 * (https://github.com/filtering-library)
 *
 * Copyright (c) 2015 Max Planck Society,
 * 				 Autonomous Motion Department,
 * 			     Institute for Intelligent Systems
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

#pragma once


#include <fl/util/traits.hpp>

#include <fl/model/process/interface/state_transition_function.hpp>

namespace fl
{

template <
    typename State,
    typename Noise,
    typename Input,
    int Id = 0
>
class AdditiveStateTransitionFunction
    : public StateTransitionFunction<State, Noise, Input, Id>,
      public internal::AdditiveNoiseModelType
{
public:
    typedef internal::AdditiveNoiseModelType Type;

    typedef StateTransitionFunction<State, Noise, Input, Id> FunctionInterface;

    /**
     * Noise model matrix \f$N_t\f$
     */
    typedef Eigen::Matrix<
                typename Noise::Scalar,
                SizeOf<State>::Value,
                SizeOf<Noise>::Value
            > NoiseMatrix;

    /**
     * Noise covariance matrix \f$N_t\f$
     */
    typedef Eigen::Matrix<
                typename Noise::Scalar,
                SizeOf<Noise>::Value,
                SizeOf<Noise>::Value
            > NoiseCovariance;

public:
    /**
     * \brief Overridable default destructor
     */
    virtual ~AdditiveStateTransitionFunction() noexcept { }

    virtual State expected_state(const State& prev_state,
                                 const Input& input) const = 0;

    virtual const NoiseMatrix& noise_matrix() const = 0;
    virtual const NoiseCovariance& noise_covariance() const = 0;

    virtual State state(const State& prev_state,
                        const Noise& noise,
                        const Input& input) const
    {
        return expected_state(prev_state, input) + noise_matrix() * noise;
    }
};

}


