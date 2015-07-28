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
 * \file process_model_interface.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__MODEL__PROCESS__PROCESS_MODEL_INTERFACE_HPP
#define FL__MODEL__PROCESS__PROCESS_MODEL_INTERFACE_HPP

#include <fl/util/traits.hpp>

namespace fl
{

template <
    typename State_,
    typename Noise_,
    typename Input_,
    int Id = 0
>
class StateTransitionFunction
    : internal::NonAdditiveNoiseModelType
{
public:
    typedef internal::NonAdditiveNoiseModelType Type;

    typedef State_ State;
    typedef Noise_ Noise;
    typedef Input_ Input;

public:
    /**
     * \brief Overridable default destructor
     */
    virtual ~StateTransitionFunction() { }

    virtual State state(const State& prev_state,
                        const Noise& noise,
                        const Input& input) const = 0;

    /**
     * \return Dimension of the state variable $\f$x\f$
     */
    virtual int state_dimension() const = 0;

    /**
     * \return Dimension of the noise term \f$w\f$
     */
    virtual int noise_dimension() const = 0;

    /**
     * \return Dimension of the input \f$u_t\f$
     */
    virtual int input_dimension() const = 0;

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











/// \todo: this needs to disappear
template <
    typename State,
    typename Noise,
    typename Input
>
class ProcessModelInterface
    : public internal::ProcessModelType
{
public:
    typedef internal::ProcessModelType ModelType;

public:
    /**
     * Sets the conditional arguments \f$x_t, u_t\f$ of \f$p(x\mid x_t, u_t)\f$
     *
     * \param delta_time    Prediction duration \f$\Delta t\f$
     * \param state         Previous state \f$x_{t}\f$
     * \param input         Control input \f$u_t\f$
     *
     * Once the conditional have been set, may either sample from this model
     * since it also represents a conditional distribution or you may map
     * a SNV noise term \f$v_t\f$ onto the distribution using
     * \c map_standard_variate(\f$v_t\f$) if implemented.
     */
    virtual void condition(const double& delta_time,
                           const State& state,
                           const Input& input = Input()) { }

    /**
     * Predicts the state conditioned on the previous state and input.
     *
     * \param delta_time    Prediction duration \f$\Delta t\f$
     * \param state         Previous state \f$x_{t}\f$
     * \param noise         Additive or non-Additive noise \f$v_t\f$
     * \param input         Control input \f$u_t\f$
     *
     * \return State \f$x_{t+1}\sim p(x\mid x_t, u_t)\f$
     */

    /// \todo have a default argument for the input, a default function which
    /// has to be implemented by the derived classes
    virtual State predict_state(double delta_time,
                                const State& state,
                                const Noise& noise,
                                const Input& input) = 0;

    /**
     * \return \f$\dim(x_t)\f$, dimension of the state
     */
    virtual constexpr int state_dimension() const = 0;

    /**
     * \return \f$\dim(v_t)\f$, dimension of the noise
     */
    virtual constexpr int noise_dimension() const = 0;

    /**
     * \return \f$\dim(u_t)\f$, dimension of the control input
     */
    virtual constexpr int input_dimension() const = 0;
};




}

#endif
