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
 * \file additive_uncorrelated_noise_model.hpp
 * \date July 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__MODEL__ADDITIVE_UNCORRELATED_NOISE_MODEL_HPP
#define FL__MODEL__ADDITIVE_UNCORRELATED_NOISE_MODEL_HPP

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

#endif
