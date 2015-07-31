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
 * \file use.hpp
 * \date July 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__UTIL__META__OPERATOR__USE_AS_HPP
#define FL__UTIL__META__OPERATOR__USE_AS_HPP

#include <type_traits>
#include <fl/util/types.hpp>
#include <fl/util/traits.hpp>

namespace fl
{

template <typename T> struct UseAs;

template <typename Model_>
struct UseAs<Additive<Model_>>
{
    typedef Model_ Model;
    typedef Additive<Model> Type;

    static_assert(IsAdditive<Model>::Value, "Model noise is not addtive");
};

template <typename Model_>
struct UseAs<AdditiveUncorrelated<Model_>>
{
    typedef Model_ Model;
    typedef AdditiveUncorrelated<Model> Type;

    static_assert(
        IsAdditiveUncorrelated<Model>::Value,
        "Model noise is not of type addtive and uncorrelated");
};

template <typename Model_>
struct UseAs<NonAdditive<Model_>>
{
    typedef Model_ Model;
    typedef NonAdditive<Model> Type;

    static_assert(
        IsNonAdditive<Model>::Value,
        "Model noise is not of type non-addtive");
};

}

#endif
