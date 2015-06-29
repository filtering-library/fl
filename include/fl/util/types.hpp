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
 * \file types.hpp
 * \date June 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__UTIL__TYPES_HPP
#define FL__UTIL__TYPES_HPP

#include <Eigen/Dense>

namespace fl
{

#if defined(FL_USE_FLOAT)
    typedef float FloatingPoint;
#elif defined(FL_USE_DOUBLE)
    typedef double FloatingPoint;
#elif defined(FL_USE_LONG_DOUBLE)
    typedef long double FloatingPoint;
#else
    typedef double FloatingPoint;
#endif

/**
 * \ingroup types
 */
typedef FloatingPoint Real;

/**
 * \internal
 */
namespace internal
{

/**
 * \internal
 * \ingroup types
 *
 * Observation model type identifier
 */
struct ObsrvModelType { };

/**
 * \internal
 * \ingroup types
 *
 * Process model type identifier
 */
struct ProcessModelType { };

/**
 * \internal
 * \ingroup types
 *
 * Adaptive model type identifier
 */
struct AdaptiveModelType { };

}

}

#endif
