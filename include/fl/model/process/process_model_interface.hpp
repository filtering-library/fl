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
 * \file process_model_interface.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__MODEL__PROCESS__PROCESS_MODEL_INTERFACE_HPP
#define FL__MODEL__PROCESS__PROCESS_MODEL_INTERFACE_HPP

namespace fl
{

template <typename State, typename Noise, typename Input>
class ProcessModelInterface
{
public:
    virtual State predict_state(double delta_time,
                                const State& state,
                                const Noise& noise,
                                const Input& input) = 0;

    virtual size_t state_dimension() const = 0;
    virtual size_t noise_dimension() const = 0;
    virtual size_t input_dimension() const = 0;
};

}

#endif
