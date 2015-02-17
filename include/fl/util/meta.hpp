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
 * \file meta.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__UTIL__META_HPP
#define FL__UTIL__META_HPP

#include <Eigen/Dense>
#include <fl/util/traits.hpp>

namespace fl
{

/**
 * \ingroup meta
 *
 * \tparam Sizes    Variadic argument containing a list of sizes. Each size
 *                  is a result of a constexpr or a const of a signed integral
 *                  type. Unknown sizes are represented by -1 or Eigen::Dynamic.
 *
 *
 * Joins the sizes within the \c Sizes argument list if all sizes are
 * non-dynamic. That is, all sizes are greater -1 (\c Eigen::Dynamic). If one of
 * the passed sizes is \c Eigen::Dynamic, the total size collapses to
 * \c Eigen::Dynamic.
 */
template <int... Sizes> struct JoinSizes;

/** 
 * \internal
 * \ingroup meta 
 * \copydoc JoinSizes
 *
 * \tparam Head     Head of the \c Sizes list of the previous recursive call
 *
 * Variadic counting recursion
 */
template <int Head, int... Sizes> struct JoinSizes<Head, Sizes...>
{
    enum : signed int
    {
        Size = IsFixed<Head>() && IsFixed<JoinSizes<Sizes...>::Size>()
                   ? Head + JoinSizes<Sizes...>::Size
                   : Eigen::Dynamic
    };
};

/**
 * \internal
 * \ingroup meta
 *
 * Terminal specialization of JoinSizes<...>
 */
template <> struct JoinSizes<> { enum: signed int { Size = 0 }; } ;

/**
 * \ingroup meta
 *
 * \brief Function form of JoinSizes for two sizes
 */
inline constexpr int join_sizes(int a, int b)
{
    return (a > Eigen::Dynamic && b > Eigen::Dynamic) ? a + b : Eigen::Dynamic;
}

/**
 * \ingroup meta
 *
 * Computes the product of LocalSize times Factor. If one of the parameters is
 * set to Eigen::Dynamic, the factor size will collapse to Eigen::Dynbamic as
 * well.
 */
template <int ... Sizes> struct ExpandSizes;

/**
 * \internal
 * \ingroup meta
 */
template <int Head, int... Sizes> struct ExpandSizes<Head, Sizes...>
{
    enum: signed int
    {
        Size = IsFixed<Head>() && IsFixed<ExpandSizes<Sizes...>::Size>()
                   ? Head * ExpandSizes<Sizes...>::Size
                   : Eigen::Dynamic
    };
};

/**
 * \internal
 * \ingroup meta
 *
 * Terminal specialization of ExpandSizes<...>
 */
template <> struct ExpandSizes<> { enum: signed int { Size = 1 }; } ;

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
template <typename First, typename...T> struct FirstTypeIn<First, T...>
{
    typedef First Type;
};

/**
 * \ingroup meta
 *
 * Represents a sequence of indices IndexSequence<0, 1, 2, 3, ...>
 *
 * This is particularly useful to expand tuples
 *
 * \tparam Indices  List of indices starting from 0
 */
template <int ... Indices> struct IndexSequence
{
    enum { Size = sizeof...(Indices) };
};

/**
 * \ingroup meta
 *
 * Creates an IndexSequence<0, 1, 2, ...> for a specified size.
 */
template <int Size, int ... Indices>
struct CreateIndexSequence
    : CreateIndexSequence<Size - 1, Size - 1, Indices...>
{ };

/**
 * \internal
 * \ingroup meta 
 *
 * Terminal specialization CreateIndexSequence
 */
template <int ... Indices>
struct CreateIndexSequence<0, Indices...>
    : IndexSequence<Indices...>
{ };

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
    enum : signed int { Size = Eigen::Dynamic };
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
{ };

/**
 * \ingroup meta
 *
 * Creates an empty \c TypeSequence<> for a \c Count = Eigen::Dynamic
 */
template <typename Type>
struct CreateTypeSequence<Eigen::Dynamic, Type>
  : TypeSequence<>
{ };

/**
 * \ingroup meta
 *
 * Same as CreateTypeSequence, however with a reversed parameter order. This is
 * an attempt to make the use of \c CreateTypeSequence more natural.
 */
template <typename Type, int Count>
struct MultipleOf
    : CreateTypeSequence<Count, Type>
{ };

/**
 * \internal
 * \ingroup meta
 *
 * Creates an empty TypeSequence for dynamic count
 */
template <typename Type>
struct MultipleOf<Type, Eigen::Dynamic>
    : CreateTypeSequence<Eigen::Dynamic, Type>
{ };

}

#endif

