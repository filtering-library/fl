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
 * \file observation_model_interface.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__MODEL__OBSERVATION__OBSERVATION_MODEL_INTERFACE_HPP
#define FL__MODEL__OBSERVATION__OBSERVATION_MODEL_INTERFACE_HPP

#include <cstdlib>

namespace fl
{

struct NoObservationParameter { };

/**
 * \ingroup observation_models
 */
template <
    typename Observation,
    typename State,
    typename Noise,
    typename Parameter = NoObservationParameter,
    int Id = 0
>
class ObservationModelInterface
{
public:
    /**
     * \param state
     * \param noise
     * \param time_step_parity - Time step parity flag
     *
     * \return
     */
    virtual Observation predict_observation(const State& state,
                                            const Noise& noise,
                                            double delta_time) = 0;


    virtual int state_dimension() const = 0;
    virtual int noise_dimension() const = 0;
    virtual int observation_dimension() const = 0;

    virtual int id() const { return Id; }

    virtual void parameter(Parameter param) {  }
    virtual Parameter parameter() const { return Parameter(); }
};


/**
 * \ingroup observation_models
 *
 * Additive noise Observation model interface
 */
template <
    typename Observation,
    typename State,
    typename Noise,
    typename Parameter = NoObservationParameter,
    int Id = 0
>
class ANObservationModelInterface
    : public ObservationModelInterface<Observation, State, Noise, Parameter, Id>
{
public:
    /**
     * \param state
     * \param time_step_parity - Time step parity flag
     *
     * \return
     */
    virtual Observation predict_observation(const State& state,
                                            double delta_time) = 0;

    virtual Noise noise_covariance_vector() const = 0;
};

}

#endif
