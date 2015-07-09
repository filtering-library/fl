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

#include <utility>

#include <fl/util/meta.hpp>
#include <fl/util/traits.hpp>
#include <fl/util/profiling.hpp>

#include <fl/exception/exception.hpp>
#include <fl/filter/filter_interface.hpp>

#include <fl/model/process/linear_state_transition_model.hpp>
#include <fl/model/observation/linear_observation_model.hpp>

namespace fl
{

/**
 * \defgroup kalman_filter Kalman Filter
 * \ingroup filters
 */

// Gaussian filter forward declaration
template <typename...> class GaussianFilter;


/**
 * \internal
 * \ingroup kalman_filter
 *
 * Traits of the Linear GaussianFilter (KalmanFilter)
 */
template <typename X, typename U, typename Y>
struct Traits<
           GaussianFilter<
               LinearStateTransitionModel<X, U>,
               LinearObservationModel<Y, X>>>
{
    typedef X State;
    typedef U Input;
    typedef Y Obsrv;
    typedef Gaussian<State> Belief;
};

/**
 * \ingroup kalman_filter
 *
 * \brief GaussianFilter resembles the Kalman filter.
 *
 * \tparam State  State type defining the state space
 * \tparam Input  Process model input type
 * \tparam Obsrv  Observation type of the linear observation Gaussian model
 *
 * The KalmanFilter type is represented by the GaussianFilter using
 * the linear Gaussian Models.
 *
 */
template <
    typename State,
    typename Input,
    typename Obsrv
>
class GaussianFilter<
          LinearStateTransitionModel<State, Input>,
          LinearObservationModel<Obsrv, State>>
    :
    /* Implement the conceptual filter interface */
    public FilterInterface<
               GaussianFilter<
                   LinearStateTransitionModel<State, Input>,
                   LinearObservationModel<Obsrv, State>>>
{
public:
    typedef LinearStateTransitionModel<State, Input> ProcessModel;
    typedef LinearObservationModel<Obsrv, State> ObservationModel;

    /**
     * \brief Represents the underlying distribution of the estimated state.
     * In the case of the Kalman filter, the distribution is a simple Gaussian
     * with the dimension of the \c State
     */
    typedef Gaussian<State> Belief;

public:
    /**
     * Creates a linear Gaussian filter (a KalmanFilter)
     *
     * \param process_model         Process model instance
     * \param obsrv_model           Obsrv model instance
     */
    GaussianFilter(const ProcessModel& process_model,
                   const ObservationModel& obsrv_model)
        : process_model_(process_model),
          obsrv_model_(obsrv_model)
    { }

    /**
     * \brief Overridable default destructor
     */
    virtual ~GaussianFilter() { }

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
    virtual void predict(const Belief& prior_belief,
                         const Input& input,
                         Belief& predicted_belief)
    {
        auto A = process_model_.dynamics_matrix();
        auto B = process_model_.input_matrix();
        auto Q = process_model_.noise_covariance();

        predicted_belief.mean(
            A * prior_belief.mean() + B * input);

        predicted_belief.covariance(
            A * prior_belief.covariance() * A.transpose() + Q);
    }

    /**
     * \copydoc FilterInterface::predict(const Belief&,
     *                                   const Input&,
     *                                   long,
     *                                   Belief&)
     */
    virtual void predict(const Belief& prior_belief,
                         const Input& input,
                         const long steps,
                         Belief& predicted_belief)
    {
        if (steps == 1)
        {
            predict(prior_belief, input, predicted_belief);
            return;
        }

        auto A = process_model_.dynamics_matrix();
        auto B = process_model_.input_matrix();
        auto Q = process_model_.noise_covariance();

        auto A_pow_k = A;
        auto sum_A_pow_i= A;
        auto sum_A_pow_i_Q_AT_pow_i = Q;

        A_pow_k.setIdentity();
        sum_A_pow_i.setZero();
        sum_A_pow_i_Q_AT_pow_i.setZero();

        for (int i = 0; i < steps; ++i)
        {
            sum_A_pow_i += A_pow_k;
            sum_A_pow_i_Q_AT_pow_i += A_pow_k * Q * A_pow_k.transpose();

            A_pow_k = A_pow_k * A;
        }

        predicted_belief.mean(
            A_pow_k * prior_belief.mean()
            + sum_A_pow_i * B * input);

        predicted_belief.covariance(
            A_pow_k * prior_belief.covariance() * A_pow_k.transpose()
            + sum_A_pow_i_Q_AT_pow_i);
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
    virtual void update(const Belief& predicted_belief,
                        const Obsrv& y,
                        Belief& posterior_belief)
    {
        auto H = obsrv_model_.sensor_matrix();
        auto R = obsrv_model_.noise_covariance();

        auto&& mean = predicted_belief.mean();
        auto&& cov_xx = predicted_belief.covariance();

        auto S = (H * cov_xx * H.transpose() + R).eval();
        auto K = (cov_xx * H.transpose() * S.inverse()).eval();

        posterior_belief.mean(mean + K * (y - H * mean));
        posterior_belief.covariance(cov_xx - K * H * cov_xx);
    }

    /**
     * \copydoc FilterInterface::predict_and_update
     */
    virtual void predict_and_update(const Belief& prior_belief,
                                    const Input& input,
                                    const Obsrv& observation,
                                    Belief& posterior_belief)
    {
        predict(prior_belief, input, posterior_belief);
        update(posterior_belief, observation, posterior_belief);
    }

    virtual Belief create_belief() const
    {
        auto belief = Belief(process_model().state_dimension());
        return belief;
    }

    ProcessModel& process_model() { return process_model_; }
    ObservationModel& obsrv_model() { return obsrv_model_; }
    const ProcessModel& process_model() const { return process_model_; }
    const ObservationModel& obsrv_model() const { return obsrv_model_; }

protected:
    ProcessModel process_model_;
    ObservationModel obsrv_model_;
};

}

#endif
