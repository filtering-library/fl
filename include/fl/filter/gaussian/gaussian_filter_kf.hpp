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
 * \file gaussian_filter_kf.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__FILTER__GAUSSIAN__GAUSSIAN_FILTER_KF_HPP
#define FL__FILTER__GAUSSIAN__GAUSSIAN_FILTER_KF_HPP

#include <map>
#include <tuple>
#include <memory>
#include <typeinfo>

#include <fl/util/meta.hpp>
#include <fl/util/traits.hpp>

#include <fl/exception/exception.hpp>
#include <fl/filter/filter_interface.hpp>

#include <fl/model/process/linear_process_model.hpp>
#include <fl/model/observation/linear_observation_model.hpp>

namespace fl
{

template <typename...> class GaussianFilter;

/**
 * Traits of the Linear GaussianFilter (KalmanFilter)
 */
template <typename State_,typename Input_,typename Obsrv_>
struct Traits<
           GaussianFilter<
               LinearGaussianProcessModel<State_, Input_>,
               LinearGaussianObservationModel<Obsrv_, State_>>>
{
    /**
     * Process model definition.
     *
     * The process model of the KalmanFilter is always the
     * \c LinearGaussianProcessModel taking a \c State and an \c Input type as
     * the only parameter types.
     */
    typedef LinearGaussianProcessModel<State_, Input_> ProcessModel;

    /**
     * Observation model definition
     *
     * The observation model of the KalmanFilter is always the
     * \c LinearGaussianObservationModel taking an \c Obsrv and a
     * \c State type as the only parameters.
     */
    typedef LinearGaussianObservationModel<
                Obsrv_, State_
            > ObservationModel;

    /**
     * Represents KalmanFilter definition
     *
     * The KalmanFilter type is represented by the GaussianFilter using
     * the linear Gaussian Models.
     */
    typedef GaussianFilter<
                LinearGaussianProcessModel<State_, Input_>,
                LinearGaussianObservationModel<Obsrv_, State_>
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
    typedef typename Traits<ObservationModel>::Obsrv Obsrv;

    /**
     * Represents the underlying distribution of the estimated state. In the
     * case of the Kalman filter, the distribution is a simple Gaussian with
     * the dimension of the \c State
     */
    typedef Gaussian<State> StateDistribution;
};

/**
 * \brief GaussianFilter resembles the Kalman filter.
 *
 * \tparam State  State type defining the state space
 * \tparam Input  Process model input type
 * \tparam Obsrv  Observation type of the linear observation Gaussian model
 *
 * The KalmanFilter type is represented by the GaussianFilter using
 * the linear Gaussian Models.
 *
 * \ingroup filters
 */
template <typename State, typename Input, typename Obsrv>
class GaussianFilter<
          LinearGaussianProcessModel<State, Input>,
          LinearGaussianObservationModel<Obsrv, State>>
    :
    /* Implement the conceptual filter interface */
    public FilterInterface<
               GaussianFilter<
                   LinearGaussianProcessModel<State, Input>,
                   LinearGaussianObservationModel<Obsrv, State>>>
{
protected:
    /** \cond INTERNAL */
    typedef GaussianFilter<
                LinearGaussianProcessModel<State, Input>,
                LinearGaussianObservationModel<Obsrv, State>
            > This;
    /** \endcond */

public:
    /* public concept interface types */
    typedef typename Traits<This>::ObservationModel ObservationModel;
    typedef typename Traits<This>::ProcessModel ProcessModel;
    typedef typename Traits<This>::StateDistribution StateDistribution;

public:
    /**
     * Creates a linear Gaussian filter (a KalmanFilter)
     *
     * \param process_model         Process model instance
     * \param obsrv_model           Obsrv model instance
     */
    GaussianFilter(const std::shared_ptr<ProcessModel>& process_model,
                   const std::shared_ptr<ObservationModel>& obsrv_model)
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
        auto&& A = (process_model_->A() * delta_time).eval();
        auto&& Q = (process_model_->covariance() * delta_time).eval();

        predicted_dist.mean(
            A * prior_dist.mean());

        predicted_dist.covariance(
            A * prior_dist.covariance() * A.transpose() + Q);
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
        auto&& R = obsrv_model_->covariance();

        auto&& mean = predicted_dist.mean();
        auto&& cov_xx = predicted_dist.covariance();

        auto&& S = (H * cov_xx * H.transpose() + R).eval();
        auto&& K = (cov_xx * H.transpose() * S.inverse()).eval();

        posterior_dist.mean(mean + K * (y - H * mean));
        posterior_dist.covariance(cov_xx - K * H * cov_xx);
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
    std::shared_ptr<ObservationModel> obsrv_model_;
};

}

#endif
