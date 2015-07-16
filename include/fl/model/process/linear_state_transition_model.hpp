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
 * \file linear_process_model.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__MODEL__PROCESS__LINEAR_GAUSSIAN_PROCESS_MODEL_HPP
#define FL__MODEL__PROCESS__LINEAR_GAUSSIAN_PROCESS_MODEL_HPP

#include <fl/util/traits.hpp>
#include <fl/distribution/gaussian.hpp>
#include <fl/model/process/interface/state_transition_function.hpp>

#include <fl/model/process/interface/state_transition_density.hpp>
#include <fl/model/process/interface/additive_state_transition_function.hpp>

namespace fl
{

/**
 * \ingroup process_models
 *
 * \brief Represents the discrete-time linear state transition model of any
 * linear or linearized system of the form
 * \f$ x_{t+1} =F_tx_t + G_tu_t + N_tv_t \f$.
 */
template <typename State_, typename Input_>
class LinearStateTransitionModel
    : public StateTransitionDensity<State_, Input_>,
      public AdditiveStateTransitionFunction<State_, State_, Input_>
{
public:
    typedef State_ State;
    typedef Input_ Input;

    typedef StateTransitionDensity<State, Input> DensityInterface;
    typedef AdditiveStateTransitionFunction<State, State, Input> AdditiveInterface;
    typedef typename AdditiveInterface::FunctionInterface FunctionInterface;


    /**
     * \brief Linear model density. The density for linear model is the Gaussian
     * over state space.
     */
    typedef Gaussian<State> Density;

    /**
     * \brief Dynamics model matrix \f$F_t\f$
     */
    typedef Eigen::Matrix<
                typename State::Scalar,
                SizeOf<State>::Value,
                SizeOf<State>::Value
            > DynamicsMatrix;

    /**
     * \brief Control input model matrix\f$G_t\f$
     */
    typedef Eigen::Matrix<
                typename Input::Scalar,
                SizeOf<State>::Value,
                SizeOf<Input>::Value
            > InputMatrix;

    /**
     * \brief Noise (effect) model matrix \f$N_t\f$
     */
    typedef typename AdditiveInterface::NoiseMatrix NoiseMatrix;

public:
    explicit LinearStateTransitionModel(int state_dim = DimensionOf<State>(),
                                        int input_dim = DimensionOf<Input>())
        : dynamics_matrix_(DynamicsMatrix::Identity(state_dim, state_dim)),
          input_matrix_(InputMatrix::Identity(state_dim, input_dim)),
          density_(state_dim),
          discretization_time_step_(1)
    {
        assert(state_dim > 0);
        assert(input_dim > 0);
    }

    /**
     * \brief Overridable default destructor
     */
    virtual ~LinearStateTransitionModel() { }

    virtual State expected_state(const State& state,
                                 const Input& input) const
    {
        return dynamics_matrix_ * state + input_matrix_ * input;
    }

    virtual Real log_probability(const State& state,
                                          const State& cond_state,
                                          const Input& cond_input) const
    {
        density_.mean(expected_state(cond_state, cond_input));

        return density_.log_probability(state);
    }

public: /* factory functions */
    virtual InputMatrix create_input_matrix() const
    {
        auto B = input_matrix();
        B.setIdentity();
        return B;
    }

    virtual DynamicsMatrix create_dynamics_matrix() const
    {
        auto A = dynamics_matrix();
        A.setIdentity();
        return A;
    }

    virtual NoiseMatrix create_noise_matrix() const
    {
        auto N = noise_matrix();
        N.setIdentity();
        return N;
    }

public: /* accessors & mutators */
    virtual const DynamicsMatrix& dynamics_matrix() const
    {
        return dynamics_matrix_;
    }

    virtual const InputMatrix& input_matrix() const
    {
        return input_matrix_;
    }

    virtual const NoiseMatrix& noise_matrix() const
    {
        return density_.square_root();
    }

    virtual const NoiseMatrix& noise_covariance() const
    {
        return density_.covariance();
    }

    virtual int state_dimension() const
    {
        return dynamics_matrix_.rows();
    }

    virtual int noise_dimension() const
    {
        return density_.square_root().cols();
    }

    virtual int input_dimension() const
    {
        return input_matrix_.cols();
    }

    virtual Real discretization_time_step() const
    {
        return discretization_time_step_;
    }

    virtual void dynamics_matrix(const DynamicsMatrix& dynamics_mat)
    {
        dynamics_matrix_ = dynamics_mat;
    }

    virtual void input_matrix(const InputMatrix& input_mat)
    {
        input_matrix_ = input_mat;
    }

    virtual void noise_matrix(const NoiseMatrix& noise_mat)
    {
        density_.square_root(noise_mat);
    }

    virtual void noise_covariance(const NoiseMatrix& noise_mat_squared)
    {
        density_.covariance(noise_mat_squared);
    }

    virtual void discretization_time_step(
        Real new_discretization_time_step)
    {
        discretization_time_step_ = new_discretization_time_step;
    }

protected:
    DynamicsMatrix dynamics_matrix_;
    InputMatrix input_matrix_;
    mutable Density density_;
    Real discretization_time_step_;
};

}

#endif
