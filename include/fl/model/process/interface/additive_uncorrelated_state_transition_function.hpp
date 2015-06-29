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
 * \file additive_uncorrelated_observation_function.hpp
 * \date June 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__MODEL__PROCESS__ADDITIVE_UNCORRELATED_STATE_TRANSITION_FUNCTION_HPP
#define FL__MODEL__PROCESS__ADDITIVE_UNCORRELATED_STATE_TRANSITION_FUNCTION_HPP

#include <fl/util/traits.hpp>

#include <fl/model/process/interface/additive_state_transition_function.hpp>

namespace fl
{

template <
    typename State,
    typename Noise,
    typename Input,
    int Id = 0
>
class AdditiveUncorrelatedStateTransitionFunction
    : public AdditiveStateTransitionFunction<State, Noise, Input, Id>
{
public:
    typedef AdditiveStateTransitionFunction<
                State, Noise, Input, Id
            > AdditiveFunctionInterface;

    typedef Eigen::DiagonalMatrix<
                typename Noise::Scalar,
                SizeOf<State>::Value
            > NoiseDiagonal;

    virtual const NoiseDiagonal& noise_matrix_diagonal() const = 0;
    virtual const NoiseDiagonal& noise_covariance_diagonal() const = 0;
};

}

#endif
