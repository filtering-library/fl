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
 * \file filter_interface_test.cpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>

#include <memory.h>

#include <fl/exception/exception.hpp>

#include "filter_interface_stubs.hpp"

/*
 * Generic FilterContext
 */
template <class FilterAlgorithm>
class FilterContext
{
public:
    typedef FilterAlgorithm Filter;

    FilterContext(Filter _filter)
        : filter_(_filter),
          dist_(1.),
          u_(0.),
          y_(2.)
    {
    }

    void predict()
    {
        filter_.predict(dist_, u_, dist_);
    }

    void update()
    {
        filter_.update(dist_, y_, dist_);
    }

    Filter& filter()
    {
        return filter_;
    }

    Filter filter_;
    typename FilterAlgorithm::Belief dist_;
    typename FilterAlgorithm::Input u_;
    typename FilterAlgorithm::Obsrv y_;
};

TEST(FilterInterface, NonTemplatedFilter)
{
    auto filter_context = FilterContext<FilterForFun>(FilterForFun());

    // predict pre-condition
    EXPECT_DOUBLE_EQ(filter_context.dist_, 1.);
    EXPECT_DOUBLE_EQ(filter_context.y_, 2.);

    EXPECT_NO_THROW(filter_context.predict());

    // predict post-condition
    EXPECT_DOUBLE_EQ(filter_context.dist_, 2.);
    EXPECT_DOUBLE_EQ(filter_context.y_, 2.);

    EXPECT_NO_THROW(filter_context.update());

    // update post-condition
    EXPECT_DOUBLE_EQ(filter_context.dist_, (2. + 2.)/2. );
    EXPECT_DOUBLE_EQ(filter_context.y_, 2.);
}

TEST(FilterInterface, TemplatedFilter)
{
    typedef FilterForMoreFun<int, long int, long long int> FilterAlgo;

    auto filter_context = FilterContext<FilterAlgo>(FilterAlgo());

    // predict pre-condition
    EXPECT_DOUBLE_EQ(filter_context.dist_, 1.);
    EXPECT_DOUBLE_EQ(filter_context.y_, 2.);

    EXPECT_NO_THROW(filter_context.predict());

    // predict post-condition
    EXPECT_DOUBLE_EQ(filter_context.dist_, 3.);
    EXPECT_DOUBLE_EQ(filter_context.y_, 2.);

    EXPECT_NO_THROW(filter_context.update());

    // update post-condition
    EXPECT_DOUBLE_EQ(filter_context.dist_, (3. + 2.)/3. );
    EXPECT_DOUBLE_EQ(filter_context.y_, 2.);
}
