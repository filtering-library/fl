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
 * @date 2014
 * @author Jan Issac (jan.issac@gmail.com)
 * Max-Planck-Institute for Intelligent Systems,
 * University of Southern California
 */

#include <gtest/gtest.h>

#include <memory>

#include <Eigen/Dense>

#include <fast_filtering/filtering_library/exception/exception.hpp>
#include <fast_filtering/filtering_library/filter/gaussian/gaussian_filter.hpp>
#include <fast_filtering/filtering_library/filter/gaussian/unscented_transform.hpp>

#include "gaussian_filter_stubs.hpp"

template <typename FilteringAlgorithm>
class FilteringContext
{
public:
    typedef fl::FilterInterface<FilteringAlgorithm> Filter;

    FilteringContext(const typename Filter::Ptr& filter)
        : filter_(filter)
    {

    }

    void predict()
    {
        filter_->predict(0.0, typename Filter::Input(), state_distr_, state_distr_);
    }

    void update()
    {
        filter_->update(typename Filter::Observation(), state_distr_, state_distr_);
    }

    typename Filter::Ptr filter()
    {
        return filter_;
    }

    typename Filter::Ptr filter_;
    typename Filter::Observation y_;
    typename Filter::StateDistribution state_distr_;
};


TEST(GaussianFilter, some_test)
{
    /* ===  Step 1: define vectorial types === */
    typedef Eigen::Matrix<double, 11,  1> State;
    typedef Eigen::Matrix<double, 100, 1> Obsrv;
    typedef Eigen::Matrix<double, 3,   1> Input;
    typedef Eigen::Matrix<double, 6,   1> StateNoise;
    typedef Eigen::Matrix<double, 100, 1> ObsrvNoise;

    /* == Step 2: define models == */
    typedef ProcessModelStub<State, StateNoise, Input> ProcessModel;
    typedef ObservationModelStub<State, Obsrv, ObsrvNoise> ObservationModel;

    /* == Step 3: define the custom filter algorithm == */
    typedef fl::GaussianFilter<
                ProcessModel,
                ObservationModel,
                fl::UnscentedTransform
            > UnscentedKalmanFilter;

    typedef fl::FilterInterface<UnscentedKalmanFilter> Filter;

    std::shared_ptr<Filter> filter =
        std::make_shared<UnscentedKalmanFilter>(
            std::make_shared<ProcessModel>(),
            std::make_shared<ObservationModel>(),
            std::make_shared<fl::UnscentedTransform>());

    typename Filter::StateDistribution state_distr;

    filter->predict(0.0, Input(), state_distr, state_distr);
    filter->update(Obsrv(), state_distr, state_distr);
    filter->predict_and_update(0., Input(), Obsrv(), state_distr, state_distr);
}

//TEST(GaussianFilter, context_test)
//{
//    /* ===  Step 1: define vectorial types === */
//    typedef Eigen::Matrix<double, 7,  1> State;
//    typedef Eigen::Matrix<double, 4, 1> Obsrv;
//    typedef Eigen::Matrix<double, 3,   1> Input;
//    typedef Eigen::Matrix<double, 6,   1> StateNoise;
//    typedef Eigen::Matrix<double, 4, 1> ObsrvNoise;

//    /* == Step 2: define models == */
//    typedef ProcessModelStub<State, StateNoise, Input> ProcessModel;
//    typedef ObservationModelStub<State, Obsrv, ObsrvNoise> ObservationModel;

//    /* == Step 3: define filter algorithm == */
//    typedef fl::GaussianFilter<
//                ProcessModel,
//                ObservationModel,
//                fl::UnscentedTransform
//            > UnscentedKalmanFilter;

//    FilteringContext<UnscentedKalmanFilter> filter_context(
//        std::make_shared<UnscentedKalmanFilter>(
//            std::make_shared<ProcessModel>(),
//            std::make_shared<ObservationModel>(),
//            std::make_shared<fl::UnscentedTransform>()));

//    filter_context.predict();
//    filter_context.update();
//}

