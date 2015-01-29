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
 * \file general_functions.hpp
 * \date January 2015
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__UTIL__MATH__GENERAL_FUNCTIONS_HPP
#define FL__UTIL__MATH__GENERAL_FUNCTIONS_HPP

#include <cmath>

namespace fl
{

/**
 * \brief Sigmoid function
 * \ingroup general_functions
 */
constexpr double sigmoid(const double& x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

/**
 * \brief log odd
 * \ingroup general_functions
 */
constexpr double logit(const double& x)
{
    return std::log(x / (1.0 - x));
}

}

#endif
