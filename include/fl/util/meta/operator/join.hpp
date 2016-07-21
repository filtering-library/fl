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
 * \file join.hpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once


#include "adaptive.hpp"
#include "multiple_of.hpp"
#include "../type_sequence.hpp"

namespace fl
{

/**
 * \ingroup meta
 *
 * \c Join represents a meta operator taking an argument pack which should be
 * unified in a specific way. The operator is defined by the following rules
 *
 * Mapping:     Join<T...> = T...
 * Associative: Join<Join<O1, O2>, O3> =  Join<O1, Join<O2, O3>> = {O1, O2, O3}
 * Affinity:    Join<Adaptive<O1, P1>, O2> = Adaptive<Join<<O1, O2>, Join<P1>>
 */
template <typename...T> struct Join;

/**
 * \internal
 * \ingroup meta
 *
 * Collapses the argument pack of the Join operator into Model<T...> or given
 * sequence within the Join<...> operator pack.
 */
template <typename...T> struct CollapseJoin;

/**
 * \internal
 * \ingroup meta
 *
 * Collapses the argument pack of an adaptive Join operator into Model<T...> or
 * given sequence within the Join<...> operator pack. Unlike CollapseJoin,
 * CollapseAdaptiveJoin results in Adaptive<Model<...>, JointTransition<...>>
 * Where JointTransition<...> repreents the joint process of all parameters
 * of the final adaptive Model<...>.
 */
template <typename...T> struct CollapseAdaptiveJoin;

/**
 * \internal
 * \ingroup meta
 *
 * Inherit from this to display the join collapse result
 */
template <typename ... S> struct ShowCompilerResult
{
  static_assert(sizeof...(S) < 0, "Showing compiler result");
};

/**
 * \internal
 * \ingroup meta
 *
 * CollapseJoin specialization which expands the type sequence by the Head
 * element assuming Head is neither a Join<...> nor a MultipleOf<P, #> operator.
 */
template <
    typename Head,
    typename...S,
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
        TypeSequence<S..., Head>,   // updated type sequence
        T...>                       // Remaining Tail
{ };

/**
 * \internal
 * \ingroup meta
 *
 * CollapseJoin specialization which branches of to an adaptive joint model.
 */
template <
    typename Head,
    typename Process,
    typename...S,
    typename...T,
    template <typename...> class Model
>
struct CollapseJoin<
           Model<>,                 // Final model type
           TypeSequence<S...>,      // Type sequence holding the pack expansion
           Adaptive<Head, Process>, // Adaptive model specification
           T...>                    // Remaining Tail
  : CollapseAdaptiveJoin<
        Model<>,
        TypeSequence<NotAdaptive<S>...>,
        TypeSequence<>,
        //typename CreateTypeSequence<sizeof...(S), NATransition>::TypeSeq,
        Adaptive<Head, Process>,
        T...>
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
* CollapseJoin specialization which branches of to an adaptive joint model.
 */
template <
    typename...S,
    typename...T,
    typename U,
    typename V,
    int Count,
    template <typename...> class Model
>
struct CollapseJoin<
           Model<>,                 // Final model type
           TypeSequence<S...>,      // Type sequence holding the pack expansion
           MultipleOf<Adaptive<U, V>, Count>,    // Current pack head element
           T...>                    // Remaining Join pack, the tail
  : CollapseAdaptiveJoin<
        Model<>,
        TypeSequence<NotAdaptive<S>...>,
        TypeSequence<>,
        //typename CreateTypeSequence<sizeof...(S), NATransition>::TypeSeq,
        MultipleOf<Adaptive<U, V>, Count>,
        T...>
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

/**
 * \internal
 * \ingroup meta
 *
 * Terminal case of Join operator collapsing for a Join of the form
 * Join<MultipleOf<P, #>> resulting in Model<MultipleOf<P, #>>
 */
template <
    typename M,
    typename Process,
    int Count,
    template <typename...> class Model
>
struct CollapseJoin<
           Model<>,
           TypeSequence<Model<MultipleOf<Adaptive<M, Process>, Count>>>>
    : CollapseAdaptiveJoin<
           Model<>,
           TypeSequence<Model<MultipleOf<Adaptive<M, Process>, Count>>>>
{ };


template <typename...> class JointTransition;

/**
 * \internal
 * \ingroup meta
 */
template <
    typename Head,
    typename Process,
    typename...S,
    typename...T,
    typename...U,
    template <typename...> class Model
>
struct CollapseAdaptiveJoin<
           Model<>,
           TypeSequence<S...>,
           TypeSequence<T...>,
           Adaptive<Head, Process>,
           U...>
  : CollapseAdaptiveJoin<
        Model<>,
        TypeSequence<S..., Head>,
        TypeSequence<T..., Process>,
        U...>
{ };

/**
 * \internal
 * \ingroup meta
 */
template <
    typename Head,
    typename...S,
    typename...T,
    typename...U,
    template <typename...> class Model
>
struct CollapseAdaptiveJoin<
           Model<>,
           TypeSequence<S...>,
           TypeSequence<T...>,
           Head,
           U...>
  : CollapseAdaptiveJoin<
        Model<>,
        TypeSequence<S..., NotAdaptive<Head>>,
        TypeSequence<T...>,
        //TypeSequence<T..., NATransition>,
        U...>
{ };

/**
 * \internal
 * \ingroup meta
 */
template <
    typename...S,
    typename...T,
    typename...U,
    typename...V,
    template <typename...> class Model
>
struct CollapseAdaptiveJoin<
           Model<>,
           TypeSequence<S...>,
           TypeSequence<T...>,
           Join<V...>,
           U...>
  : CollapseAdaptiveJoin<
        Model<>,
        TypeSequence<S...>,
        TypeSequence<T...>,
        V...,
        U...>
{ };

/**
 * \internal
 * \ingroup meta
 */
template <
    typename...S,
    typename...T,
    typename...U,
    typename V,
    int Count,
    template <typename...> class Model
>
struct CollapseAdaptiveJoin<
           Model<>,
           TypeSequence<S...>,
           TypeSequence<T...>,
           MultipleOf<V, Count>,
           U...>
  : CollapseAdaptiveJoin<
        Model<>,
        TypeSequence<S...>,
        TypeSequence<T...>,
        Model<MultipleOf<V, Count>>,
        U...>
{ };


/**
 * \internal
 * \ingroup meta
 */
template <
    typename...S,
    typename...T,
    typename...U,
    typename V,
    typename Process,
    int Count,
    template <typename...> class Model
>
struct CollapseAdaptiveJoin<
           Model<>,
           TypeSequence<S...>,
           TypeSequence<T...>,
           MultipleOf<Adaptive<V, Process>, Count>,
           U...>
  : CollapseAdaptiveJoin<
        Model<>,
        TypeSequence<S...>,
        TypeSequence<T...>,
        Adaptive<
            Model<MultipleOf<V, Count>>,
            JointTransition<MultipleOf<Process, Count>>
        >,
        U...>
{ };

/**
 * \internal
 * \ingroup meta
 *
 * Terminal case of CollapseAdaptiveJoin
 */
template <
    typename...S,
    typename...T,
    template <typename...> class Model
>
struct CollapseAdaptiveJoin<
           Model<>,
           TypeSequence<S...>,
           TypeSequence<T...>>
//        : ShowCompilerResult<Adaptive<Model<S...>, JointTransition<T...>>>
{
    /**
     * \brief Final type outcome of the Join Operator
     */
    typedef Adaptive<Model<S...>, JointTransition<T...>> Type;

//    static_assert(sizeof...(S) > 1,
//                  "Join<A, B, ...> operator must take 2 or more operands");
};

/**
 * \internal
 * \ingroup meta
 *
 * Terminal case of CollapseAdaptiveJoin for single JointTransition<>
 */
template <
    typename...S,
    typename...T,
    template <typename...> class Model
>
struct CollapseAdaptiveJoin<
           Model<>,
           TypeSequence<S...>,
           TypeSequence<JointTransition<T...>>>
//        : ShowCompilerResult<Adaptive<Model<S...>, JointTransition<T...>>>
{
    /**
     * \brief Final type outcome of the Join Operator
     */
    typedef Adaptive<Model<S...>, JointTransition<T...>> Type;
};

/**
 * \internal
 * \ingroup meta
 *
 * Terminal case of CollapseAdaptiveJoin for singe JointTransition<> and
 * single Model<>
 */
template <
    typename...S,
    typename...T,
    template <typename...> class Model
>
struct CollapseAdaptiveJoin<
           Model<>,
           TypeSequence<Model<S...>>,
           TypeSequence<JointTransition<T...>>>
//        : ShowCompilerResult<Adaptive<Model<S...>, JointTransition<T...>>>
{
    /**
     * \brief Final type outcome of the Join Operator
     */
    typedef Adaptive<Model<S...>, JointTransition<T...>> Type;
};


/**
 * \internal
 * \ingroup meta
 *
 * Terminal case of CollapseAdaptiveJoin for MultipleOf<Adaptive<M, Process>, #>
 */
template <
    typename M,
    typename Process,
    int Count,
    template <typename...> class Model
>
struct CollapseAdaptiveJoin<
           Model<>,
           TypeSequence<Model<MultipleOf<Adaptive<M, Process>, Count>>>>
//        : ShowCompilerResult<Adaptive<Model<S...>, JointTransition<T...>>>
{
    /**
     * \brief Final type outcome of the Join Operator
     */
    typedef Adaptive<
                Model<MultipleOf<M, Count>>,
                JointTransition<MultipleOf<Process, Count>>
            > Type;
};

}


