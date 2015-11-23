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
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once


#include <fl/util/traits.hpp>

#include <fl/model/additive_noise_model.hpp>
#include <fl/model/observation/interface/observation_function.hpp>

namespace fl
{

template <
    typename Obsrv,
    typename State,
    typename NoiseDensity,
    int Id = 0
>
class AdditiveObservationFunction
    : public ObservationFunction<Obsrv, State, typename NoiseDensity::Variate, Id>,
      public AdditiveNoiseModel<NoiseDensity>
{
public:
    typedef typename NoiseDensity::Variate Noise;
    typedef internal::AdditiveNoiseModelType Type;
    typedef ObservationFunction<Obsrv, State, Noise, Id> FunctionInterface;
    typedef AdditiveNoiseModel<NoiseDensity> AdditiveInterface;

    using AdditiveInterface::noise_matrix;

public:
    /**
     * \brief Overridable default destructor
     */
    virtual ~AdditiveObservationFunction() noexcept { }

    /**
     * Evaluates the model function \f$y = h(x, w)\f$ where \f$x\f$ is the state
     * and \f$w\sim {\cal N}(0, 1)\f$ is a white noise parameter. Put
     * differently, \f$y = h(x, w)\f$ is a sample from the conditional model
     * distribution \f$p(y \mid x)\f$.
     *
     * \param state         The state variable \f$x\f$
     * \param noise         The noise term \f$w\f$
     * \param delta_time    Prediction time
     */
    virtual Obsrv expected_observation(const State& state) const = 0;

    Obsrv observation(const State& state, const Noise& noise) const override
    {
        return expected_observation(state) + noise_matrix() * noise;
    }
};

}


