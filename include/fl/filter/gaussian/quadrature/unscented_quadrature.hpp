/*
 * This is part of the FL library, a C++ Bayesian filtering library
 * (https://github.com/filtering-library)
 *
 * Copyright (c) 2014 Jan Issac (jan.issac@gmail.com)
 * Copyright (c) 2014 Manuel Wuthrich (manuel.wuthrich@gmail.com)
 *
 * This Source Code Form is subject to the terms of the MIT License (MIT).
 * A copy of the license can be found in the LICENSE file distributed with this
 * source code.
 */

/**
 * \file unscented_quadrature.hpp
 * \date July 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once


#include <fl/filter/gaussian/transform/unscented_transform.hpp>
#include <fl/filter/gaussian/quadrature/sigma_point_quadrature.hpp>

namespace fl
{

class UnscentedQuadrature
    : public SigmaPointQuadrature<UnscentedTransform>
{
public:
    /**
     * Creates a UnscentedQuadrature
     *
     * \param alpha     UT Scaling parameter alpha (distance to the mean)
     * \param beta      UT Scaling parameter beta  (2.0 is optimal for Gaussian)
     * \param kappa     UT Scaling parameter kappa (higher order parameter)
     */
    UnscentedQuadrature(Real alpha = 1.0, Real beta = 2., Real kappa = 0.)
        : SigmaPointQuadrature<UnscentedTransform>(
              UnscentedTransform(alpha, beta, kappa))
    { }
};

}
