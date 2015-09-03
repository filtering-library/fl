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
 * \file assertions.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once


#include <assert.h>
#include <type_traits>

/**
 * \ingroup macros
 * \internal
 *
 * This variadic macro performs a compile time derivation assertion. The first
 * argument is the derived type which is being tested whether it implements a
 * base type. The base type is given as the second argument list.
 *
 * __VA_ARGS__ was used as a second parameter to enable passing template
 * specialization to the macro.
 *
 * Note: The macro requires <em>derived_type</em> to be a single worded type. In
 *       case of a template specialization, use a typedef.
 */
#define static_assert_base(derived_type, ...)\
    static_assert(( \
            std::is_base_of<__VA_ARGS__, derived_type>::value), \
            #derived_type " must derive from " #__VA_ARGS__);

#define static_assert_dynamic_sized(matrix)\
    static_assert(matrix::SizeAtCompileTime == Eigen::Dynamic,\
                  "Calling a fixed-size function on a dynamic-size one.");

#define static_assert_const_sized(matrix)\
    static_assert(matrix::SizeAtCompileTime != Eigen::Dynamic,\
                  "Calling a dynamic size function on a fixed-size one.");


