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
 * \file gaussian_filter_linear.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once


#include <utility>

#include <fl/util/meta.hpp>
#include <fl/util/traits.hpp>
#include <fl/util/profiling.hpp>

#include <fl/exception/exception.hpp>
#include <fl/filter/filter_interface.hpp>

#include <fl/model/process/linear_state_transition_model.hpp>
#include <fl/model/observation/linear_gaussian_observation_model.hpp>

namespace fl
{

/**
 * \defgroup linear_gaussian_filter Linear Gaussian Filter (aka Kalman Filter)
 * \ingroup filters
 */

// Gaussian filter forward declaration
template <typename...> class GaussianFilter;


/**
 * \internal
 * \ingroup linear_gaussian_filter
 *
 * Traits of the Linear GaussianFilter (KalmanFilter)
 */
template <typename LinearStateTransitionModel, typename LinearObservationModel>
struct Traits<
           GaussianFilter<
               LinearStateTransitionModel,
               LinearObservationModel>>
{
    typedef typename LinearStateTransitionModel::State State;
    typedef typename LinearStateTransitionModel::Input Input;
    typedef typename LinearObservationModel::Obsrv Obsrv;
    typedef Gaussian<State> Belief;
};

/**
 * \ingroup linear_gaussian_filter
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
    typename LinearStateTransitionModel,
    typename LinearObservationModel
>
class GaussianFilter<
          LinearStateTransitionModel,
          LinearObservationModel>
    :
    /* Implement the conceptual filter interface */
    public
    FilterInterface<
        GaussianFilter<
            typename ForwardLinearModelOnly<LinearStateTransitionModel>::Type,
            typename ForwardLinearModelOnly<LinearObservationModel>::Type>>
{
public:
    typedef typename LinearStateTransitionModel::State State;
    typedef typename LinearStateTransitionModel::Input Input;
    typedef typename LinearObservationModel::Obsrv Obsrv;

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
    GaussianFilter(const LinearStateTransitionModel& process_model,
                   const LinearObservationModel& obsrv_model)
        : process_model_(process_model),
          obsrv_model_(obsrv_model)
    { }

    /**
     * \brief Overridable default destructor
     */
    virtual ~GaussianFilter() noexcept { }

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

        auto mean = predicted_belief.mean();
        auto cov_xx = predicted_belief.covariance();

        auto S = (H * cov_xx * H.transpose() + R).eval();
        auto K = (cov_xx * H.transpose() * S.inverse()).eval();

        posterior_belief.mean(mean + K * (y - H * mean));
        posterior_belief.covariance(cov_xx - K * H * cov_xx);
    }

    virtual Belief create_belief() const
    {
        auto belief = Belief(process_model().state_dimension());
        return belief;
    }

    virtual std::string name() const
    {
        return "GaussianFilter<"
                + this->list_arguments(
                            process_model().name(),
                            obsrv_model().name())
                + ">";
    }

    virtual std::string description() const
    {
        return "Linear Gaussian filter (the Kalman Filter) with"
                + this->list_descriptions(
                            process_model().description(),
                            obsrv_model().description());
    }

    LinearStateTransitionModel& process_model()
    {
        return process_model_;
    }

    LinearObservationModel& obsrv_model()
    {
        return obsrv_model_;
    }

    const LinearStateTransitionModel& process_model() const
    {
        return process_model_;
    }

    const LinearObservationModel& obsrv_model() const
    {
        return obsrv_model_;
    }

protected:
    /** \cond internal */
    LinearStateTransitionModel process_model_;
    LinearObservationModel obsrv_model_;
    /** \endcond */
};

}


