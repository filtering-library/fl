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
 * \file options_argument.hpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once


namespace fl
{

/**
 * \ingroup traits
 *
 * \brief FilterOptions define a set of options which are used to specialize
 * filter algorithms.
 */
enum FilterOptions : unsigned int
{
    NoOptions = 0,                  /**< Represents empty option set */
    AdditiveProcessNoise = 1 << 0,  /**< Forces additive process noise */
    AdditiveObsrvNoise   = 1 << 1,  /**< Forces additive observation noise */
    FactorizeParams = 1 << 2        /**< Specifies param. state factorization */
};

enum ModelOptions : bool
{
    IsNotAdaptive = 0,
    IsAdaptive = 1
};

/**
 * \ingroup meta
 * Represents any number of available options as defined in #FilterOptions
 */
template <int T> struct Options { enum : signed int { Value = T }; };

/**
 * \ingroup meta
 * Combines multiple options into one. This allows to specify the options
 * listed in an arbitrary number
 */
template <int ... T> struct CombineOptions;

/**
 * \internal
 * \ingroup meta
 *
 * Combines the first option within the option pack with the previous one.
 */
template <int Value, int H, int ... T>
struct CombineOptions<Value, H, T...> : CombineOptions<Value | H, T...> { };

/**
 * \internal
 * \ingroup meta
 *
 * Terminal case CombineOptions
 */
template <int Value>
struct CombineOptions<Value> { typedef fl::Options<Value> Options; };

/**
 * \internal
 * \ingroup meta
 *
 * No Option case
 */
template <>
struct CombineOptions<> { typedef fl::Options<NoOptions> Options; };

template <int ... T>
struct MakeOptions : CombineOptions<T...> { };

}


