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
#include <memory>

#include <fast_filtering/utils/traits.hpp>

#include <fast_filtering/filtering_library/exception/exception.hpp>
#include <fast_filtering/filtering_library/filter/filter_interface.hpp>
#include <fast_filtering/filtering_library/filter/gaussian/point_set_gaussian.hpp>
#include <fast_filtering/filtering_library/filter/gaussian/point_set_transform.hpp>


namespace fl
{

// Forward declarations
template <class ProcessModel, class ObservationModel> class GaussianFilter;

/**
 * GaussianFilter Traits
 */
template <class ProcessModel, class ObservationModel>
struct Traits<GaussianFilter<ProcessModel, ObservationModel>>
{
    typedef std::shared_ptr<GaussianFilter<ProcessModel, ObservationModel>> Ptr;

    typedef double State;
    typedef double Observation;

    typedef PointSetGaussian<Eigen::VectorXd, -1> StateDistribution;
    typedef PointSetGaussian<Eigen::VectorXd, -1> StateNoiseDistribution;

    typedef PointSetTransform<
                StateDistribution,
                StateDistribution
            > Transform;
};

/**
 * GaussianFilter represents all filters based on Gaussian distributed systems.
 * This includes the Kalman Filter and filters using non-linear models such as
 * Sigma Point Kalman Filter family.
 *
 * \tparam ProcessModel
 * \tparam ObservationModel
 *
 * \ingroup filters
 */
template<class ProcessModel, class ObservationModel>
class GaussianFilter:
        public FilterInterface<
            GaussianFilter<ProcessModel, ObservationModel>
        >
{
public:
    typedef GaussianFilter<ProcessModel, ObservationModel> This;

    typedef typename Traits<This>::Ptr Ptr;
    typedef typename Traits<This>::State State;
    typedef typename Traits<This>::Observation Observation;    
    typedef typename Traits<This>::StateDistribution StateDistribution;
    typedef typename Traits<This>::Transform Transform;

protected:
    /** \cond INTERNAL */
    typedef typename Traits<This>::StateNoiseDistribution StateNoiseDistribution;
    /** \endcond */

public:

    /**
     * Creates a Gaussian filter
     *
     * @param process_model         Process model instance
     * @param observation_model     Observation model instance
     * @param poit_set_transform    Point set tranfrom such as the unscented
     *                              transform
     */
    GaussianFilter(
            const std::shared_ptr<ProcessModel>& process_model,
            const std::shared_ptr<ObservationModel>& observation_model,
            const std::shared_ptr<Transform>& point_set_transform)
        : process_model_(process_model),
          observation_model_(observation_model),
          point_set_transform_(point_set_transform)
    {

    }

    /**
     * \copydoc FilterInterface::predict(...)
     */
    virtual void predict(double delta_time,
                         const StateDistribution& prior_dist,
                         StateDistribution& predicted_dist)
    {

    }

    /**
     * \copydoc FilterInterface::update(...)
     */
    virtual void update(const StateDistribution& predicted_dist,
                        const Observation& observation,
                        StateDistribution& posterior_dist)
    {

    }

protected:
    std::shared_ptr<ProcessModel> process_model_;
    std::shared_ptr<ObservationModel> observation_model_;
    std::shared_ptr<Transform> point_set_transform_;

private:
    StateDistribution X_;
};

}

#endif
