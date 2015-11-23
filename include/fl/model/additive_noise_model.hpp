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
 * \file additive_noise_model.hpp
 * \date July 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once


#include <fl/util/types.hpp>
#include <fl/util/traits.hpp>

#include <Eigen/Dense>

namespace fl
{

template <typename NoiseDensity>
class AdditiveNoiseModel
    : private internal::AdditiveNoiseModelType
{
public:
    typedef internal::AdditiveNoiseModelType Type;

    /**
     * Noise model matrix \f$N_t\f$
     */
    typedef typename NoiseDensity::SecondMoment NoiseMatrix;
public:
    /**
     * \brief Overridable default destructor
     */
    virtual ~AdditiveNoiseModel() noexcept { }

    virtual NoiseMatrix noise_matrix() const = 0;
    virtual NoiseMatrix noise_covariance() const = 0;
};

}


