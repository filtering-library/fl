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

#include <memory.h>

#include <fl/util/traits.hpp>
#include <fl/exception/exception.hpp>
#include <fl/filter/filter_interface.hpp>


/*
 * FilterForFun forward declaration
 */
class FilterForFun;

namespace fl
{
/*
 * FilterForFun Traits
 */
template <> struct Traits<FilterForFun>
{
    typedef double State;
    typedef double Input;
    typedef double Observation;
    typedef double StateDistribution;
    typedef std::shared_ptr<FilterForFun> Ptr;
};
}

/*
 * FilterForFun Stub Filter
 */
class FilterForFun:
        public fl::FilterInterface<FilterForFun>
{
public:
    typedef FilterForFun This;

    typedef typename fl::Traits<This>::Ptr Ptr;
    typedef typename fl::Traits<This>::State State;
    typedef typename fl::Traits<This>::Input Input;
    typedef typename fl::Traits<This>::Observation Observation;
    typedef typename fl::Traits<This>::StateDistribution StateDistribution;

    virtual void predict(double delta_time,
                         const Input& input,
                         const StateDistribution& prior_dist,
                         StateDistribution& predicted_dist)
    {
        predicted_dist = (prior_dist * 2) * delta_time;
    }

    virtual void update(const StateDistribution& predicted_dist,
                        const Observation& observation,
                        StateDistribution& posterior_dist)
    {
        posterior_dist = (predicted_dist + observation) / 2.;
    }

    virtual void predict_and_update(double delta_time,
                                    const Input& input,
                                    const Observation& observation,
                                    const StateDistribution& prior_dist,
                                    StateDistribution& posterior_dist)
    {
        predict(delta_time, input, prior_dist, posterior_dist);
        update(observation, posterior_dist, posterior_dist);
    }
};


// == FilterForMoreFun Stub Filter ========================================== //

/*
 * FilterForMoreFun forward declaration
 */
template <typename A, typename B, typename C> class FilterForMoreFun;

namespace fl
{
/*
 * FilterForMoreFun Traits
 */
template <typename A, typename B, typename C>
struct Traits<FilterForMoreFun<A, B, C>>
{
    typedef double State;
    typedef double Input;
    typedef double Observation;
    typedef double StateDistribution;
    typedef std::shared_ptr<FilterForMoreFun<A, B, C>> Ptr;
};
}

/*
 * FilterForMoreFun Stub Filter
 */
template <typename A, typename B, typename C>
class FilterForMoreFun:
        public fl::FilterInterface<FilterForMoreFun<A, B, C>>
{
public:
    typedef FilterForMoreFun<A, B, C> This;

    typedef typename fl::Traits<This>::Ptr Ptr;
    typedef typename fl::Traits<This>::State State;
    typedef typename fl::Traits<This>::Input Input;
    typedef typename fl::Traits<This>::Observation Observation;
    typedef typename fl::Traits<This>::StateDistribution StateDistribution;

    virtual void predict(double delta_time,
                         const Input& input,
                         const StateDistribution& prior_dist,
                         StateDistribution& predicted_dist)
    {
        predicted_dist = (prior_dist * 3) * delta_time;
    }

    virtual void update(const StateDistribution& predicted_dist,
                        const Observation& observation,
                        StateDistribution& posterior_dist)
    {
        posterior_dist = (predicted_dist + observation) / 3.;
    }

    virtual void predict_and_update(double delta_time,
                                    const Input& input,
                                    const Observation& observation,
                                    const StateDistribution& prior_dist,
                                    StateDistribution& posterior_dist)
    {
        predict(delta_time, input, prior_dist, posterior_dist);
        update(observation, posterior_dist, posterior_dist);
    }
};
