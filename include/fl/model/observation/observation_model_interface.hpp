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
 * \file observation_model_interface.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__MODEL__OBSERVATION__OBSERVATION_MODEL_INTERFACE_HPP
#define FL__MODEL__OBSERVATION__OBSERVATION_MODEL_INTERFACE_HPP

#include <fl/util/traits.hpp>

namespace fl
{

/**
 * \ingroup observation_models
 * \interface ObservationModelInterface
 *
 *
 * Represents the generic observation model interface of the model function
 * \f$h(x, w, \theta)\f$ where \f$x\f$ is the state,
 * \f$w\sim {\cal N}(0, 1)\f$ is a white noise term, and \f$\theta\f$ the
 * model variable parameters.
 *
 * \tparam Obsrv  Measurement type \f$y = h(x, w, \theta)\f$
 * \tparam State        State variate \f$x\f$
 * \tparam Noise        White noise variate term \f$w\sim {\cal N}(0, I)\f$
 * \tparam Id           Model id number
 */
template <
    typename Obsrv,
    typename State,
    typename Noise,
    int Id = 0
>
class ObservationModelInterface
    : public internal::ObsrvModelType
{
public:
    typedef internal::ObsrvModelType ModelType;

    /**
     * Evaluates the model function \f$y = h(x, w)\f$ where \f$x\f$ is the state
     * and \f$w\sim {\cal N}(0, 1)\f$ is a white noise parameter. Put
     * differently, \f$y = h(x, w)\f$ is a sample from the conditional model
     * distribution \f$p(y \mid x)\f$.
     *
     * \param state         The state variable \f$x\f$
     * \param noise         The noise term \f$w\f$
     * \param delta_time    Prediction time
     */
    virtual Obsrv predict_obsrv(const State& state,
                                const Noise& noise,
                                double delta_time) = 0;

    /**
     * \return Dimension of the state variable $\f$x\f$
     */
    virtual int state_dimension() const = 0;

    /**
     * \return Dimension of the noise term \f$w\f$
     */
    virtual int noise_dimension() const = 0;

    /**
     * \return Dimension of the measurement \f$h(x, w)\f$
     */
    virtual int obsrv_dimension() const = 0;

    /**
     * \return Model id number
     *
     * In case of multiple sensors of the same kind, this function returns the
     * id of the individual model.
     */
    virtual int id() const { return Id; }

    /**
     * Sets the model id
     *
     * \param new_id    Model's new ID
     */
    virtual void id(int) { /* const ID */ }
};

/**
 * \ingroup observation_models
 *
 * \brief This is the observation model interface with additive noise. The noise
 * is represented as uncorrelated covariance matrix \f$R\f$. The model function
 * is of the form
 *
 * \f$ y = h(x, \theta) + w \f$
 *
 * with noise term \f$w \sim {\cal N}(0, R)\f$. Since the noise is assumed to
 * be uncorrelated white noise, the \f$R\f$ matrix has a diagonal form \f$diag(
 * \sigma_1, \sigma_2, \ldots, \sigma_M)\f$. The representation of \f$R\f$ is a
 * vector containing the diagonal elements.
 *
 *
 * \copydoc ObservationModelInterface
 */
template <
    typename Obsrv,
    typename State,
    typename Noise,
    int Id = 0
>
class ANObservationModelInterface
    : public ObservationModelInterface<Obsrv, State, Noise, Id>
{
public:
    /**
     * Evaluates the model function \f$y = h(x)\f$ where \f$x\f$ is the state.
     *
     * \param state         The state variable \f$x\f$
     * \param delta_time    Prediction time
     */
    virtual Obsrv predict_obsrv(const State& state,
                                double delta_time) = 0;

    /**
     * \return Noise covariance \f$R = diag(\sigma_1, \sigma_2, \ldots,
     * \sigma_M)\f$ as a column vector containing the \f$\sigma_i\f$. The noise
     * covariance vector type is same as the noise variate type.
     */
    virtual Noise noise_covariance_vector() const = 0;
};

template <
    typename Obsrv_,
    typename State_,
    typename Noise_,
    int Id = 0
>
class ObservationFunction
{
public:
    typedef Obsrv_ Obsrv;
    typedef State_ State;
    typedef Noise_ Noise;

    /**
     * Evaluates the model function \f$y = h(x, w)\f$ where \f$x\f$ is the state
     * and \f$w\sim {\cal N}(0, 1)\f$ is a white noise parameter. Put
     * differently, \f$y = h(x, w)\f$ is a sample from the conditional model
     * distribution \f$p(y \mid x)\f$.
     *
     * \param state         The state variable \f$x\f$
     * \param noise         The noise term \f$w\f$
     * \param delta_time    Prediction time
     */
    virtual Obsrv observation(const State& state,
                              const Noise& noise) const = 0;

    /**
     * \return Dimension of the state variable $\f$x\f$
     */
    virtual int state_dimension() const = 0;

    /**
     * \return Dimension of the noise term \f$w\f$
     */
    virtual int noise_dimension() const = 0;

    /**
     * \return Dimension of the measurement \f$h(x, w)\f$
     */
    virtual int obsrv_dimension() const = 0;

    /**
     * \return Model id number
     *
     * In case of multiple sensors of the same kind, this function returns the
     * id of the individual model.
     */
    virtual int id() const { return Id; }

    /**
     * Sets the model id
     *
     * \param new_id    Model's new ID
     */
    virtual void id(int) { /* const ID */ }
};

}

#endif
