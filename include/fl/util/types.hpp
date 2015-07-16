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
 * \brief Common floating point type. The default type of Real is \c double.
 *        This type is used throughout the entire ::fl library.
 *
 * In order or use other basic floating point types please compile with one of
 * the following defines:
 *
 *  - \c FL_USE_FLOAT:       defines fl::Real as \c float
 *  - \c FL_USE_DOUBLE:      defines fl::Real as \c double (default)
 *  - \c FL_USE_LONG_DOUBLE: defines fl::Real as \c long double
 */
typedef FloatingPoint Real;

/**
 * \ingroup types
 */
template <typename Model_> struct Additive
{
    typedef Model_ Model;
};

/**
 * \ingroup types
 */
template <typename Model_> struct AdditiveUncorrelated
{
    typedef Model_ Model;
};

/**
 * \ingroup types
 */
template <typename Model_> struct NonAdditive
{
    typedef Model_ Model;
};

/**
 * \internal
 */
namespace internal
{

/**
 * \internal
 * \ingroup types
 *
 * \brief Observation model type identifier
 */
struct ObsrvModelType { };

/**
 * \internal
 * \ingroup types
 *
 * \brief Process model type identifier
 */
struct ProcessModelType { };

/**
 * \internal
 * \ingroup types
 *
 * \brief Adaptive model type identifier
 */
struct AdaptiveModelType { };

/**
 * \internal
 * \ingroup types
 *
 * \brief Represents the base type of any model with additive noise term
 * \f$ x_{t+1} = f(x_t) + v_t\f$ while \f$v_t\f$ is the additive noise.
 */
struct AdditiveModelType { };

/**
 * \internal
 * \ingroup types
 *
 * \brief Represents the base type of any model with additive uncorrelated
 * Gaussian white noise term in \f$ x_{t+1} = f(x_t) + v_t\f$ while \f$v_t\f$ is
 * the additive noise with \f$v_t \sim {\cal N}(v_t; 0, Q_t)\f$. Here, the
 * covariance matrix has a diagonal form \f$Q_t = \text{diag}(q_1, q_2, \ldots,
 * q_n)\f$ and \f$n\f$ is the dimension of \f$v_t \in \mathbb{R}^n\f$.
 */
struct AdditiveUncorrelatedModelType { };

/**
 * \internal
 * \ingroup types
 *
 * \brief Represents the base type of any model with non-additive noise term
 * \f$ x_{t+1} = f(x_t, v_t) \f$ while \f$v_t\f$ is the additive noise.
 */
struct NonAdditiveModelType { };

}

}

#endif
