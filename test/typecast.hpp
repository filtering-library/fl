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
 * \file test_typing.cpp
 * \date June 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <fl/util/meta.hpp>

#pragma once


namespace fl
{

struct NoParameter { };

template <typename Param = NoParameter>
struct DynamicTest { typedef Param Parameter; };

template <typename Param = NoParameter>
struct StaticTest  { typedef Param Parameter; };

template <int, typename> struct TestSize;

template <int Size, typename Param>
struct TestSize<Size, StaticTest<Param>>
{
    enum : signed int { Value = Size };
};

template <int Size, typename Param>
struct TestSize<Size, DynamicTest<Param>>
{
    enum : signed int { Value = Eigen::Dynamic };
};

template <int Key_, typename TypeName>
struct IntegerTypePair
{
    enum : signed int { Key = Key_ };
    typedef TypeName Type;
};

template <int Key, typename CurrentPair, typename ... Pairs>
struct IntegerTypeMapImp
    : IntegerTypeMapImp<Key, Pairs...>
{ };

template <int Key, typename TypeName, typename ... Pairs>
struct IntegerTypeMapImp<Key, IntegerTypePair<Key, TypeName>, Pairs...>
{
    typedef TypeName Type;
};

template <typename ... Pairs>
struct IntegerTypeMap
{
    template <int Key> struct Select: IntegerTypeMapImp<Key, Pairs...> { };
};

}


