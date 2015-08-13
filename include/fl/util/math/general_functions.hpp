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
#include <fl/util/types.hpp>
#include <fl/util/math/special_functions.hpp>

namespace fl
{

/**
 * \brief Sigmoid function
 * \ingroup general_functions
 */
constexpr Real sigmoid(const Real& x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

/**
 * \brief log odd
 * \ingroup general_functions
 */
constexpr Real logit(const Real& x)
{
    return std::log(x / (1.0 - x));
}



inline long timesteps(Real discretization_time_step,
               Real delta_time)
{
    return std::round(delta_time/discretization_time_step);

//    // constexpr return function version
//    return (delta_time/discretization_time_step) -
//           - int(delta_time/discretization_time_step) > 0.5 ?
//           int(delta_time/discretization_time_step) + 1 :
//           int(delta_time/discretization_time_step);
}


/**
 * \ingroup general_functions
 * \return True if d is within the specified epsilon bounds
 */
inline bool check_epsilon_bounds(Real d, Real epsilon)
{
    return std::fabs(d) < epsilon;
}

/**
 * \ingroup general_functions
 *
 * \return converts a standard normal variate into a uniformly distributed
 * variate u
 */
inline Real normal_to_uniform(Real snv)
{
    static constexpr Real sqrt_of_2 = std::sqrt(Real(2));

    Real u = (1 + std::erf(snv / sqrt_of_2)) / Real(2);
    return u;
}

/**
 * \ingroup general_functions
 *
 * \return converts a standard normal variate into a uniformly distributed
 * variate u
 */
inline Real uniform_to_normal(Real u)
{
    static constexpr Real sqrt_of_2 = std::sqrt(Real(2));

    Real snv = fl::erfinv(Real(2) * u - 1) * sqrt_of_2;
    return snv;
}

}

#endif
