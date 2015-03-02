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
 * \file type_sequence.hpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__UTIL__META__TYPE_SEQUENCE_HPP
#define FL__UTIL__META__TYPE_SEQUENCE_HPP

namespace fl
{

/**
 * \ingroup meta
 *
 * Provides access to the the first type element in the specified variadic list
 */
template <typename...T> struct FirstTypeIn;

/**
 * \internal
 * \ingroup meta
 *
 * Implementation of FirstTypeIn
 */
template <typename First, typename...T>
struct FirstTypeIn<First, T...>
{
    typedef First Type;
};

/**
 * \ingroup meta
 *
 * Meta type defined in terms of a sequence of types
 */
template <typename ... T>
struct TypeSequence
{
    enum : signed int { Size = sizeof...(T) };
};

/**
 * \internal
 * \ingroup meta
 *
 * Empty TypeSequence
 */
template <> struct TypeSequence<>
{
    enum : signed int { Size = -1 };
};

/**
 * \ingroup meta
 *
 * Creates a \c TypeSequence<T...> by expanding the specified \c Type \c Count
 * times.
 */
template <int Count, typename Type, typename...T>
struct CreateTypeSequence
    : CreateTypeSequence<Count - 1, Type, Type, T...>
{ };

/**
 * \internal
 * \ingroup meta
 *
 * Terminal type of CreateTypeSequence
 */
template <typename Type, typename...T>
struct CreateTypeSequence<1, Type, T...>
    : TypeSequence<Type, T...>
{
    typedef TypeSequence<Type, T...> TypeSeq;
};

/**
 * \internal
 * \ingroup meta
 *
 * Terminal type of CreateTypeSequence for Count = 0 resulting in an empty
 * type sequence
 */
template <typename Type>
struct CreateTypeSequence<0, Type>
    : TypeSequence<>
{
    typedef TypeSequence<> TypeSeq;
};

/**
 * \ingroup meta
 *
 * Creates an empty \c TypeSequence<> for a \c Count = Eigen::Dynamic
 */
template <typename Type>
struct CreateTypeSequence<-1, Type>
  : TypeSequence<>
{ };


}

#endif
