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

#ifndef FL__FILTER__FILTER_INTERFACE_HPP
#define FL__FILTER__FILTER_INTERFACE_HPP

#include <fast_filtering/utils/traits.hpp>

namespace fl
{

/**
 * \class FilterInterface
 * \brief Common filter interface
 *
 * FilterInterface is the base of each filter.
 *
 * \note Techincal note: The interface employs static polymorphism using CTRP.
 * That is, the base class takes the derived type as its template argument.
 * This interface functions dispatch is then performed directly insteady of
 * dynamically using vtable lookups.
 *
 * \ingroup Filter
 */
template <typename Derived>
class FilterInterface
{
public:


    /**
     * Predicts the distribution over the state give a delta time in seconds
     *
     * \param delta_time        Delta time to predict for
     * \param prior_dist        Prior state distribution
     * \param predicted_dist    Predicted state distribution
     */
    template <typename StateDistribution>
    void predict(double delta_time,
                 const StateDistribution& prior_dist,
                 StateDistribution& predicted_dist)
    {
        static_cast<Derived*>(this)->predict(delta_time,
                                             prior_dist,
                                             predicted_dist);
    }

    /**
     * Updates a predicted state given an observation
     *
     * \param predicted_dist
     * \param observation
     * \param posterior_dist
     */
    template <typename StateDistribution, typename Observation>
    void update(const StateDistribution& predicted_dist,
                const Observation& observation,
                StateDistribution& posterior_dist)
    {
        static_cast<Derived*>(this)->update(predicted_dist,
                                            observation,
                                            posterior_dist);
    }
};

}

#endif
