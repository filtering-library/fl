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
 * \file additive_uncorrelated_observation_function.hpp
 * \date June 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once


#include <fl/util/traits.hpp>

#include <fl/model/additive_uncorrelated_noise_model.hpp>
#include <fl/model/observation/interface/additive_observation_function.hpp>

namespace fl
{

template <
    typename Obsrv,
    typename State,
    typename Noise,
    int Id = 0
>
class AdditiveUncorrelatedObservationFunction
    : public AdditiveObservationFunction<Obsrv, State, Noise, Id>,
      public AdditiveUncorrelatedNoiseModel<Noise>,
      private internal::AdditiveUncorrelatedNoiseModelType
{
public:
    /**
     * \brief Overridable default destructor
     */
    virtual ~AdditiveUncorrelatedObservationFunction() noexcept { }
};

}


