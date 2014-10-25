/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Max-Planck-Institute for Intelligent Systems,
 *                     University of Southern California
 *    Jan Issac (jan.issac@gmail.com)
 *    Manuel Wuthrich (manuel.wuthrich@gmail.com)
 *
 *  All rights reserved.
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
 * @date 10/21/2014
 * @author Jan Issac (jan.issac@gmail.com)
 * @author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * Max-Planck-Institute for Intelligent Systems,
 * University of Southern California
 */

#ifndef FL__FILTER__GAUSSIAN__GAUSSIAN_FILTER_HPP
#define FL__FILTER__GAUSSIAN__GAUSSIAN_FILTER_HPP

#include <map>
#include <tuple>

#include <fast_filtering/filtering_library/filter/gaussian/point_set_gaussian.hpp>
#include <fast_filtering/filtering_library/exception/exception.hpp>
#include <fast_filtering/filtering_library/filter/filter_interface.hpp>

namespace fl
{

template <class ProcessModel, class ObservationModel>
/**
 * @brief The GaussianFilter class
 */
class GaussianFilter:
        public FilterInterface<
            GaussianFilter<ProcessModel, ObservationModel>
        >
{
public:
//    typedef typename ProcessModel::State State;
//    typedef typename ObservationModel::Observation Observation;

    typedef PointSetGaussian<double, 0, double> StateDistribution;

    virtual void predict(double delta_time,
                         const StateDistribution& prior_dist,
                         StateDistribution& predicted_dist)
    {

    }

    virtual void update(const StateDistribution& predicted_dist,
                        const Observation& observation,
                        StateDistribution& posterior_dist)
    {

    }
};

// this went too far...
//template <class ProcessModel, class ObservationModel>
//class GaussianFilter
//{

//};

//template<
//    template <typename, typename...> class ProcessModel,
//    template <typename, typename...> class ObservationModel,
//    typename State,
//    typename Observation,
//    typename... ProcessModelArgs,
//    typename... ObservationModelArgs
//>
//class GaussianFilter
//      <
//        ProcessModel<State, ProcessModelArgs...>,
//        ObservationModel<Observation, ObservationModelArgs...>
//      >
//      : public FilterInterface
//        <
//            GaussianFilter
//            <
//              ProcessModel<State, ProcessModelArgs...>,
//              ObservationModel<Observation, ObservationModelArgs...>
//            >
//        >
//{
//public:
//    //typedef PointSetGaussian<double, 0, double> StateDistribution;
//    typedef double StateDistribution;
//    typedef double Observation;

//    virtual void predict(double delta_time,
//                         const StateDistribution& prior_dist,
//                         StateDistribution& predicted_dist)
//    {

//    }

//    virtual void update(const StateDistribution& predicted_dist,
//                        const Observation& observation,
//                        StateDistribution& posterior_dist)
//    {

//    }
//};

}

#endif
