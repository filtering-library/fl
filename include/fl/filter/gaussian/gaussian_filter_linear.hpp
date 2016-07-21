/*
 * This is part of the fl library, a C++ Bayesian filtering library
 * (https://github.com/filtering-library)
 *
 * Copyright (c) 2015 Max Planck Society,
 * 				 Autonomous Motion Department,
 * 			     Institute for Intelligent Systems
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

#include <fl/model/transition/linear_transition.hpp>
#include <fl/model/sensor/linear_gaussian_sensor.hpp>

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
template <typename LinearTransition, typename LinearSensor>
struct Traits<
           GaussianFilter<
               LinearTransition,
               LinearSensor>>
{
    typedef typename LinearTransition::State State;
    typedef typename LinearTransition::Input Input;
    typedef typename LinearSensor::Obsrv Obsrv;
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
    typename LinearTransition,
    typename LinearSensor
>
class GaussianFilter<
          LinearTransition,
          LinearSensor>
    :
    /* Implement the conceptual filter interface */
    public
    FilterInterface<
        GaussianFilter<
            typename ForwardLinearModelOnly<LinearTransition>::Type,
            typename ForwardLinearModelOnly<LinearSensor>::Type>>
{
public:
    typedef typename LinearTransition::State State;
    typedef typename LinearTransition::Input Input;
    typedef typename LinearSensor::Obsrv Obsrv;

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
     * \param transition         Process model instance
     * \param sensor           Obsrv model instance
     */
    GaussianFilter(const LinearTransition& transition,
                   const LinearSensor& sensor)
        : transition_(transition),
          sensor_(sensor)
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
        auto A = transition_.dynamics_matrix();
        auto B = transition_.input_matrix();
        auto Q = transition_.noise_covariance();

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
        auto H = sensor_.sensor_matrix();
        auto R = sensor_.noise_covariance();

        auto mean = predicted_belief.mean();
        auto cov_xx = predicted_belief.covariance();

        auto S = (H * cov_xx * H.transpose() + R).eval();
        auto K = (cov_xx * H.transpose() * S.inverse()).eval();

        posterior_belief.mean(mean + K * (y - H * mean));
        posterior_belief.covariance(cov_xx - K * H * cov_xx);
    }

    virtual Belief create_belief() const
    {
        auto belief = Belief(transition().state_dimension());
        return belief;
    }

    virtual std::string name() const
    {
        return "GaussianFilter<"
                + this->list_arguments(
                            transition().name(),
                            sensor().name())
                + ">";
    }

    virtual std::string description() const
    {
        return "Linear Gaussian filter (the Kalman Filter) with"
                + this->list_descriptions(
                            transition().description(),
                            sensor().description());
    }

    LinearTransition& transition()
    {
        return transition_;
    }

    LinearSensor& sensor()
    {
        return sensor_;
    }

    const LinearTransition& transition() const
    {
        return transition_;
    }

    const LinearSensor& sensor() const
    {
        return sensor_;
    }

protected:
    /** \cond internal */
    LinearTransition transition_;
    LinearSensor sensor_;
    /** \endcond */
};

}


