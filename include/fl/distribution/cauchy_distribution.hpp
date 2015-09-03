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
 * \file cauchy_distribution.hpp
 * \date August 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once


#include <Eigen/Dense>

#include <fl/util/types.hpp>
#include "t_distribution.hpp"

namespace fl
{

/**
 * \ingroup distributions
 *
 * \brief CauchyDistribution represents a multivariate cauchy distribution. It
 * is a special case of student's t-distribution and is equal to
 * \f$t_1(\mu, \Sigma)\f$.
 */
template <typename Variate>
class CauchyDistribution
    : public TDistribution<Variate>
{
public:
    /**
     * Creates a dynamic or fixed size multivariate cauchy distribution
     *
     * \param dimension Dimension of the distribution. The default is defined by
     *                  the dimension of the variable type \em Vector. If the
     *                  size of the Variate at compile time is fixed, this will
     *                  be adapted. For dynamic-sized Variable the dimension is
     *                  initialized to 0.
     */
    explicit CauchyDistribution(int dim = DimensionOf<Variate>())
        : TDistribution<Variate>(Real(1), dim)
    { }
};

}
