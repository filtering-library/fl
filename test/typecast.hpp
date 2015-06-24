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
 * \file test_typing.cpp
 * \date June 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>

namespace fl
{

struct DynamicTest { };
struct StaticTest { };

template <int, typename> struct TestSize;

template <int Size>
struct TestSize<Size, StaticTest>
{
    enum : signed int { Value = Size };
};

template <int Size>
struct TestSize<Size, DynamicTest>
{
    enum : signed int { Value = Eigen::Dynamic };
};

}

