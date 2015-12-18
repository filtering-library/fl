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
 * \file additive_uncorrelated_state_transition_function.hpp
 * \date June 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once


#include <fl/util/traits.hpp>

#include <fl/model/process/interface/additive_state_transition_function.hpp>

namespace fl
{

template <
    typename State,
    typename Input,
    int Id = 0
>
class AdditiveUncorrelatedStateTransitionFunction
    : public AdditiveStateTransitionFunction<State, State, Input, Id>
{
public:
    typedef AdditiveStateTransitionFunction<
                State, State, Input, Id
            > AdditiveFunctionInterface;

    typedef Eigen::DiagonalMatrix<
                typename State::Scalar,
                SizeOf<State>::Value
            > NoiseDiagonal;

public:
    /**
     * \brief Overridable default destructor
     */
    virtual ~AdditiveUncorrelatedStateTransitionFunction() noexcept { }

    virtual const NoiseDiagonal& noise_diagonal_matrix() const = 0;
    virtual const NoiseDiagonal& noise_diagonal_covariance() const = 0;


    /* TODO: override state() function using noise_diagonal_matrix() */
};

}
