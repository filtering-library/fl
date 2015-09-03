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
 * \file index_sequence.hpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once


namespace fl
{

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

}


