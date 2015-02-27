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
 * \file join.hpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__UTIL__META__OPERATOR__JOIN_HPP
#define FL__UTIL__META__OPERATOR__JOIN_HPP

#include "multiple_of.hpp"
#include "../type_sequence.hpp"

namespace fl
{


/**
 * \c Join represents a meta operator taking an argument pack which should be
 * unified in a specific way. The operator is defined by the following axioms
 *
 * - Pack of Join<P1, P2, ...>: {P1, P2, ...}
 * - Positive pack:             sizeof...(T) > 0 for Join<T...>
 * - Nesting:                   Join<P1, Join<P2, P3>> = {P1, P2, P3}
 * - Comm.                      Join<P1, Join<P2, P3>> = Join<Join<P1, P2>, P3>
 * - MultipleOf operator:       Join<MultipleOf<P, #>>
 *
 * \ingroup meta
 */
template <typename...T> struct Join;

/**
 * \internal
 * \ingroup meta
 *
 * Collapes the argument pack of the Join operator into Model<T...> of ay given
 * Join<...> operator.
 */
template <typename...T> struct CollapseJoin;

/**
 * \internal
 * \ingroup meta
 *
 * CollapseJoin specialization which expands the type sequence by the Head
 * element assuming Head is neither a Join<...> nor a MultipleOf<P, #> operator.
 */
template <
    typename...S,
    typename Head,
    typename...T,
    template <typename...> class Model
>
struct CollapseJoin<
           Model<>,                 // Final model type
           TypeSequence<S...>,      // Type sequence holding the pack expansion
           Head,                    // Current non-operator pack head element
           T...>                    // Remaining Tail
  : CollapseJoin<
        Model<>,                    // Final model type
        TypeSequence<Head, S...>,   // updated type sequence
        T...>                       // Remaining Tail
{ };

/**
 * \internal
 * \ingroup meta
 *
 * CollapseJoin specialization which unifies a Join<...> pack Head into the
 * outer Join pack, i.e.
 *
 * Join<T1..., Join<T2...>, T3 > = Join<T1..., T2..., T3...>
 */
template <
    typename...S,
    typename...T,
    typename...U,
    template <typename...> class Model
>
struct CollapseJoin<
           Model<>,                 // Final model type
           TypeSequence<S...>,      // Type sequence holding the pack expansion
           Join<T...>,              // Current pack head element
           U...>                    // Remaining Tail
  : CollapseJoin<
        Model<>,                    // Final model type
        TypeSequence<S...>,         // Type sequence holding the pack expansion
        T...,                       // Extracted pack from inner Join operator
        U...>                       // Remaining Tail
{ };

/**
 * \internal
 * \ingroup meta
 *
 * CollapseJoin specialization which translates a MultipleOf<P,#> operator,
 * at the head of the pack, into Model<MultipleOf<P, #>>, i.e
 *
 * Join<T1..., MultipleOf<P, 10>, T2...> =
 * Join<T1..., Model<MultipleOf<P, 10>>, T2...>.
 */
template <
    typename...S,
    typename...T,
    typename U,
    int Count,
    template <typename...> class Model
>
struct CollapseJoin<
           Model<>,                 // Final model type
           TypeSequence<S...>,      // Type sequence holding the pack expansion
           MultipleOf<U, Count>,    // Current pack head element
           T...>                    // Remaining Join pack, the tail
  : CollapseJoin<
        Model<>,                    // Final model type
        TypeSequence<S...>,         // Type sequence holding the pack expansion
        Model<MultipleOf<U, Count>>,// Redefine head in terms of Model<>
        T...>                       // Remaining Tail
{ };


/**
 * \internal
 * \ingroup meta
 *
 * Terminal case of Join operator collapsing. The final result of Join<T...>
 * is Model<U...> where U are all the expanded non operator types.
 */
template <
    typename...S,
    template <typename...> class Model
>
struct CollapseJoin<
           Model<>,
           TypeSequence<S...>>
{
    /**
     * \brief Final type outcome of the Join Operator
     */
    typedef Model<S...> Type;

    static_assert(sizeof...(S) > 1,
                  "Join<A, B, ...> operator must take 2 or more operands");
};

/**
 * \internal
 * \ingroup meta
 *
 * Terminal case of Join operator collapsing for a Join of the form
 * Join<MultipleOf<P, #>> resulting in Model<MultipleOf<P, #>>
 */
template <
    typename M,
    int Count,
    template <typename...> class Model
>
struct CollapseJoin<Model<>, TypeSequence<Model<MultipleOf<M, Count>>>>
{
    typedef Model<MultipleOf<M, Count>> Type;

    static_assert(Count > 0 || Count == -1,
                  "Cannot expand Join<MultipleOf<M, Count>> operator on M. "
                  "Count must be positive or set to -1 for the dynamic size "
                  "case.");
};

}

#endif
