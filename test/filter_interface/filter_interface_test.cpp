/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Max-Planck-Institute for Intelligent Systems,
 *                     University of Southern California
 *    Jan Issac (jan.issac@gmail.com)
 *    Manuel Wuthrich (manuel.wuthrich@gmail.com)
 *
 *
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

/**
 * \date 2014
 * \author Jan Issac (jan.issac@gmail.com)
 * Max-Planck-Institute for Intelligent Systems,
 * University of Southern California
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
    typedef std::shared_ptr<FilterAlgorithm> Filter;

    FilterContext(Filter _filter)
        : filter_(_filter),
          dist_(1.),
          u_(0.),
          y_(2.)
    {
    }

    void predict()
    {
        filter_->predict(1.0, u_, dist_, dist_);
    }

    void update()
    {
        filter_->update(dist_, y_, dist_);
    }

    Filter& filter()
    {
        return filter_;
    }

    Filter filter_;
    typename FilterAlgorithm::StateDistribution dist_;
    typename FilterAlgorithm::Input u_;
    typename FilterAlgorithm::Obsrv y_;
};

TEST(FilterInterface, NonTemplatedFilter)
{
    typedef FilterForFun FilterAlgo;

    FilterContext<FilterAlgo> filter_context =
        FilterContext<FilterAlgo>(std::make_shared<FilterAlgo>());

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

//TEST(FilterInterface, TemplatedFilter)
//{
//    typedef FilterForMoreFun<int, long int, long long int> FilterAlgo;

//    FilterContext<FilterAlgo> filter_context(std::make_shared<FilterAlgo>());

//    // predict pre-condition
//    EXPECT_DOUBLE_EQ(filter_context.dist_, 1.);
//    EXPECT_DOUBLE_EQ(filter_context.y_, 2.);

//    EXPECT_NO_THROW(filter_context.predict());

//    // predict post-condition
//    EXPECT_DOUBLE_EQ(filter_context.dist_, 3.);
//    EXPECT_DOUBLE_EQ(filter_context.y_, 2.);

//    EXPECT_NO_THROW(filter_context.update());

//    // update post-condition
//    EXPECT_DOUBLE_EQ(filter_context.dist_, (3. + 2.)/3. );
//    EXPECT_DOUBLE_EQ(filter_context.y_, 2.);
//}
