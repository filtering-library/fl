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
 * \file multiple_of.hpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once


#include <type_traits>

#include <fl/util/meta/operator/not_adaptive.hpp>
#include <fl/util/meta/operator/forward_adaptive.hpp>

namespace fl
{

/**
 * \ingroup meta
 *
 * Same as CreateTypeSequence, however with a reversed parameter order. This is
 * an attempt to make the use of \c CreateTypeSequence more natural. It also
 * allows dynamic sizes if needed.
 */
template <typename T, int Count_>
struct MultipleOf
    //: CreateTypeSequence<Count, typename ForwardAdaptive<T>::Type>
{
    //typedef typename ForwardAdaptive<T>::Type Type;

    enum : signed int { Count = Count_ };
    typedef T Type;

    MultipleOf(const Type& instance, int instance_count = Count)
        : instance(instance),
          count(instance_count)
    { }

    Type instance;
    int count;
};

}


