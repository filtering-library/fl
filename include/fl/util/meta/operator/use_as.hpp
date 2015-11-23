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
 * \file use.hpp
 * \date July 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once


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


