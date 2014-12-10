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
 *  (C) Copyright Greg Colvin and Beman Dawes 1998, 1999.
 *  Copyright (c) 2001-2008 Peter Dimov
 *
 *  Distributed under the MIT Software License. (See accompanying file LICENSE
 *  or copy at http://opensource.org/licenses/MIT)
 */

/**
 * @date 10/21/2014
 * @author Jan Issac (jan.issac@gmail.com)
 * Max-Planck-Institute for Intelligent Systems,
 * University of Southern California
 */

#ifndef FL__FILTER__FILTER_INTERFACE_HPP
#define FL__FILTER__FILTER_INTERFACE_HPP

#include <fast_filtering/utils/traits.hpp>

namespace fl
{

/**
 * \interface FilterInterface
 *
 * \brief Common filter interface
 *
 *
 *
 * FilterInterface represents the common interface of all filters. Each filter
 * must implement the two filter steps, predict/propagate and update.
 * The filtered state, observation and the state distribution are subject to the
 * underlying filtering algorithm and therefore may differ from one to another.
 * This interface, provides these types of the actual algorithm, such as the
 * the State, Observation, and StateDistribution types. Therefore, the traits
 * of each filter implememntation must provide the following types
 *
 * Required Types        | Description | Requirements
 * --------------------- | ----------- | -
 * \c State              | Used State type | -
 * \c Input              | Process control input type | -
 * \c Observation        | Used Observation type | -
 * \c StateDistribution  | Distribution type over the state | must implement ff::Moments
 * \c Ptr                | Shared pointer of the derived type | must specialize std::shared_ptr<>
 */
template <typename Derived>
class FilterInterface
{
public:
    /**
     * \brief Ptr is the shared pointer type of the filter specialization
     *
     * The filter specialization traits must provide the shared pointer type
     */
    typedef typename Traits<Derived>::Ptr Ptr;

    /**
     * \brief State type provided by the filter specialization
     *
     * The filter specialization traits must provide the \c State type. The
     * state type is commonly either provided by the process model or the filter
     * self.
     */
    typedef typename Traits<Derived>::State State;

    /**
     * \brief Input control type provided by the filter specialization
     *
     * The filter specialization traits must provide the \c Input control type.
     * The \c Input type is commonly either provided by the process model or the
     * filter self.
     */
    typedef typename Traits<Derived>::Input Input;

    /**
     * \brief Observation type provided by the filter specialization
     *
     * The filter specialization traits must provide the \c Observation type.
     * The \c Observation type is commonly provided by the measurement
     * model.
     */
    typedef typename Traits<Derived>::Observation Observation;

    /**
     * \brief StateDistribution type uses by the filter specialization.
     *
     * The filter specialization uses a suitable state distribution. By
     * convension, the StateDistribution must implement the Moments interface
     * which provides the first two moments.
     */
    typedef typename Traits<Derived>::StateDistribution StateDistribution;

    /**
     * Predicts the distribution over the state give a delta time in seconds
     *
     * \param delta_time        Delta time of prediction
     * \param input             Control input argument
     * \param prior_dist        Prior state distribution     
     * \param predicted_dist    Predicted state distribution
     */
    virtual void predict(double delta_time,
                         const Input& input,
                         const StateDistribution& prior_dist,
                         StateDistribution& predicted_dist) = 0;

    /**
     * Updates a predicted state given an observation
     *
     * \param predicted_dist    Predicted state distribution
     * \param observation       Latest observation
     * \param posterior_dist    Updated posterior state distribution
     */
    virtual void update(const Observation& observation,
                        const StateDistribution& predicted_dist,
                        StateDistribution& posterior_dist) = 0;

    /**
     * Predicts the state distribution for a given delta time and subsequently
     * updates the prediction using a measurement.
     *
     * @param delta_time
     * @param input
     * @param observation
     * @param prior_dist
     * @param posterior_dist
     */
    virtual void predict_and_update(double delta_time,
                                    const Input& input,
                                    const Observation& observation,
                                    const StateDistribution& prior_dist,
                                    StateDistribution& posterior_dist) = 0;
};

}

#endif
