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
 * \struct JoinSizes
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
 * \sa JoinSizes
 * \copydoc JoinSizes
 *
 * \tparam Head     Head of the \c Sizes list of the previous recursive call
 *
 * Variadic counting recursion
 */
template <int Head, int... Sizes> struct JoinSizes<Head, Sizes...>
{
    enum
    {
        Size = IsFixed<Head>()
                   ? Head + JoinSizes<Sizes...>::Size
                   : Eigen::Dynamic
    };
};

/**
 * \internal
 *
 * \ingroup meta
 * \sa JoinSizes
 *
 * Recursion termination
 */
template <> struct JoinSizes<> { enum { Size = 0 }; } ;

}

#endif

