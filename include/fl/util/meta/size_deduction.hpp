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
 * \file size_deduction.hpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once


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
        Value = IsFixed<Head>() && IsFixed<JoinSizes<Sizes...>::Size>()
                   ? Head + JoinSizes<Sizes...>::Size
                   : Eigen::Dynamic,
        Size = Value
    };
};

/**
 * \internal
 * \ingroup meta
 *
 * Terminal specialization of JoinSizes<...>
 */
template <> struct JoinSizes<> { enum: signed int { Size = 0, Value = 0 }; } ;

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
        Value = IsFixed<Head>() && IsFixed<ExpandSizes<Sizes...>::Size>()
                   ? Head * ExpandSizes<Sizes...>::Size
                   : Eigen::Dynamic,
        Size = Value
    };
};

/**
 * \internal
 * \ingroup meta
 *
 * Terminal specialization of ExpandSizes<...>
 */
template <> struct ExpandSizes<> { enum: signed int { Size = 1, Value = 1 }; } ;

}


