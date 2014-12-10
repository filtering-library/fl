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
 * Max-Planck-Institute for Intelligent Systems,
 * University of Southern California
 */

#ifndef FL__FILTER__GAUSSIAN__GAUSSIAN_FILTER_KF_HPP
#define FL__FILTER__GAUSSIAN__GAUSSIAN_FILTER_KF_HPP

#include <map>
#include <tuple>
#include <memory>

#include <fast_filtering/utils/meta.hpp>
#include <fast_filtering/utils/traits.hpp>

#include <fast_filtering/filtering_library/exception/exception.hpp>
#include <fast_filtering/filtering_library/filter/filter_interface.hpp>

#include <fast_filtering/models/process_models/linear_process_model.hpp>
#include <fast_filtering/models/observation_models/linear_observation_model.hpp>

namespace fl
{

// Forward declarations
template <typename...> class GaussianFilter;

/**
 * Traits of the Linear GaussianFilter (KalmanFilter)
 */
template <typename State_,typename Input_,typename Observation_>
struct Traits<
           GaussianFilter<
               ff::LinearGaussianProcessModel<State_, Input_>,
               ff::LinearGaussianObservationModel<Observation_, State_>>>
{
    /**
     * Process model definition.
     *
     * The process model of the KalmanFilter is always the
     * \c LinearGaussianProcessModel taking a \c State and an \c Input type as
     * the only parameter types.
     */
    typedef ff::LinearGaussianProcessModel<State_, Input_> ProcessModel;

    /**
     * Observation model definition
     *
     * The observation model of the KalmanFilter is always the
     * \c LinearGaussianObservationModel taking an \c Observation and a
     * \c State type as the only parameters.
     */
    typedef ff::LinearGaussianObservationModel<Observation_, State_> ObsrvModel;

    /**
     * Represents KalmanFilter definition
     *
     * The KalmanFilter type is represented by the GaussianFilter using
     * the linear Gaussian Models.
     */
    typedef GaussianFilter<
                ff::LinearGaussianProcessModel<State_, Input_>,
                ff::LinearGaussianObservationModel<Observation_, State_>
             > Filter;

    /*
     * Required concept (interface) types
     *
     * - Ptr
     * - State
     * - Input
     * - Observation
     * - StateDistribution
     */
    typedef std::shared_ptr<Filter> Ptr;
    typedef typename Traits<ProcessModel>::State State;
    typedef typename Traits<ProcessModel>::Input Input;
    typedef typename Traits<ObsrvModel>::Observation Observation;

    /**
     * Represents the underlying distribution of the estimated state. In the
     * case of the Kalman filter, the distribution is a simple Gaussian with
     * the dimension of the \c State
     */
    typedef ff::Gaussian<State> StateDistribution;

    /** \cond INTERNAL */
    /**
     * \brief KalmanGain Matrix
     */
    typedef Eigen::Matrix<
                typename StateDistribution::Scalar,
                State::RowsAtCompileTime,
                Observation::RowsAtCompileTime
            > KalmanGain;
    /** \endcond */
};

/**
 * \brief GaussianFilter resembles the Kalman filter.
 *
 * \tparam State_       State type defining the state space
 * \tparam Input_       Process model input type
 * \tparam Observation_ Observation type of the linear observation Gaussian model
 *
 * The KalmanFilter type is represented by the GaussianFilter using
 * the linear Gaussian Models.
 *
 * \ingroup filters
 */
template <typename State, typename Input, typename Obsrv>
class GaussianFilter<
          ff::LinearGaussianProcessModel<State, Input>,
          ff::LinearGaussianObservationModel<Obsrv, State>>
    :
    /* Implement the conceptual filter interface */
    public FilterInterface<
               GaussianFilter<
                   ff::LinearGaussianProcessModel<State, Input>,
                   ff::LinearGaussianObservationModel<Obsrv, State>>>
{
protected:
    /** \cond INTERNAL */
    typedef GaussianFilter<
                ff::LinearGaussianProcessModel<State, Input>,
                ff::LinearGaussianObservationModel<Obsrv, State>
             > This;

    typedef typename Traits<This>::KalmanGain KalmanGain;
    /** \endcond */

public:
    /* public concept interface types */
    typedef typename Traits<This>::ObsrvModel ObsrvModel;
    typedef typename Traits<This>::ProcessModel ProcessModel;
    typedef typename Traits<This>::StateDistribution StateDistribution;

public:
    /**
     * Creates a linear Gaussian filter (a KalmanFilter)
     *
     * @param process_model         Process model instance
     * @param obsrv_model           Obsrv model instance
     */
    GaussianFilter(const std::shared_ptr<ProcessModel>& process_model,
                   const std::shared_ptr<ObsrvModel>& obsrv_model)
        : process_model_(process_model),
          obsrv_model_(obsrv_model) { }

    /**
     * \copydoc FilterInterface::predict
     *
     * KalmanFilter prediction step
     *
     * Given the following matrices
     *
     *  - \f$ A \f$: Dynamics Matrix
     *  - \f$ B \f$: Dynamics Noise Covariance
     *
     * and the current state distribution
     * \f$ {\cal N}(x_t\mid \hat{x}_t, \hat{\Sigma}_{t}) \f$,
     *
     * the prediction steps is the discrete linear mapping
     *
     * \f$ \bar{x}_{t} =  A \hat{x}_t\f$ and
     *
     * \f$ \bar{\Sigma}_{t} = A\hat{\Sigma}_{t}A^T + Q \f$
     */
    virtual void predict(double delta_time,
                         const Input& input,
                         const StateDistribution& prior_dist,
                         StateDistribution& predicted_dist)
    {
        auto&& A = process_model_->A();
        auto&& Q = delta_time * process_model_->Covariance();

        predicted_dist.Mean(
            A * prior_dist.Mean());

        predicted_dist.Covariance(
            A * prior_dist.Covariance() * A.transpose() + Q);
    }

    /**
     * \copydoc FilterInterface::update
     *
     * Given the following matrices
     *
     *  - \f$ H \f$: Sensor Matrix
     *  - \f$ R \f$: Sensor Noise Covariance
     *
     * and the current predicted state distribution
     * \f$ {\cal N}(x_t\mid \bar{x}_t, \bar{\Sigma}_{t}) \f$,
     *
     * the update steps is the discrete linear mapping
     *
     * \f$ \hat{x}_{t+1} =  \bar{x}_t + K (y - H\bar{x}_t)\f$ and
     *
     * \f$ \hat{\Sigma}_{t} = (I - KH) \bar{\Sigma}_{t}\f$
     *
     * with the KalmanGain
     *
     * \f$ K = \bar{\Sigma}_{t}H^T (H\bar{\Sigma}_{t}H^T+R)^{-1}\f$.
     */
    virtual void update(const Obsrv& y,
                        const StateDistribution& predicted_dist,
                        StateDistribution& posterior_dist)
    {
        auto&& H = obsrv_model_->H();
        auto&& R = obsrv_model_->Covariance();
        auto&& cov_xx = predicted_dist.Covariance();
        auto S = H * cov_xx * H.transpose() + R;

        KalmanGain K = cov_xx * H.transpose() * S.inverse();

        posterior_dist.Mean(
            predicted_dist.Mean() + K * (y - H * predicted_dist.Mean()));

        posterior_dist.Covariance(cov_xx - K * H * cov_xx);
    }

    /**
     * \copydoc FilterInterface::predict_and_update
     */
    virtual void predict_and_update(double delta_time,
                                    const Input& input,
                                    const Obsrv& observation,
                                    const StateDistribution& prior_dist,
                                    StateDistribution& posterior_dist)
    {
        predict(delta_time, input, prior_dist, posterior_dist);
        update(observation, posterior_dist, posterior_dist);
    }

protected:
    std::shared_ptr<ProcessModel> process_model_;
    std::shared_ptr<ObsrvModel> obsrv_model_;
};

}

#endif
