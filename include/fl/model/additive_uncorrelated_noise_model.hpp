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
 * \file additive_uncorrelated_noise_model.hpp
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
class AdditiveUncorrelatedNoiseModel
    : private internal::AdditiveUncorrelatedNoiseModelType
{
public:
    typedef typename NoiseDensity::SecondMoment NoiseMatrix;
public:
    virtual NoiseMatrix noise_diagonal_matrix() const = 0;
    virtual NoiseMatrix noise_diagonal_covariance() const = 0;
};

}


