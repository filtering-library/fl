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


