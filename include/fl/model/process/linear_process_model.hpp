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
#include <fl/model/process/interface/process_model_interface.hpp>

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
template <typename State, typename Input>
class LinearStateTransitionModel
    : public StateTransitionDensity<State, Input>,
      public AdditiveStateTransitionFunction<State, State, Input>
{
public:
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
                State::SizeAtCompileTime,
                State::SizeAtCompileTime
            > DynamicsMatrix;

    /**
     * \brief Control input model matrix\f$G_t\f$
     */
    typedef Eigen::Matrix<
                typename Input::Scalar,
                State::SizeAtCompileTime,
                Input::SizeAtCompileTime
            > InputMatrix;

    /**
     * \brief Noise (effect) model matrix \f$N_t\f$
     */
    typedef typename Density::SecondMoment NoiseMatrix;

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
    virtual DynamicsMatrix dynamics_matrix() const
    {
        return dynamics_matrix_;
    }

    virtual InputMatrix input_matrix() const
    {
        return input_matrix_;
    }

    virtual NoiseMatrix noise_matrix() const
    {
        return density_.square_root();
    }

    virtual NoiseMatrix noise_matrix_squared() const
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

    virtual void noise_matrix_squared(const NoiseMatrix& noise_mat_squared)
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








// Forward declarations
template <typename State, typename Input_> class LinearGaussianProcessModel;

/**
 * Linear Gaussian process model traits. This trait definition contains all
 * types used internally within the linear model. Additionally, it provides the
 * types needed externally to use the model.
 */
template <typename State_, typename Input_>
struct Traits<LinearGaussianProcessModel<State_, Input_>>
{
    typedef Gaussian<State_> GaussianBase;

    typedef State_ State;
    typedef Input_ Input;
    typedef typename Traits<GaussianBase>::Scalar Scalar;
    typedef typename Traits<GaussianBase>::SecondMoment SecondMoment;
    typedef typename Traits<GaussianBase>::StandardVariate Noise;

    typedef ProcessModelInterface<State, Noise, Input> ProcessModelBase;

    typedef Eigen::Matrix<Scalar,
                          State::SizeAtCompileTime,
                          State::SizeAtCompileTime> DynamicsMatrix;
};

/**
 * \ingroup process_models
 * \todo correct input parameter
 */
template <typename State_, typename Input_ = Eigen::Matrix<double, 1, 1>>
class LinearGaussianProcessModel:
    public Traits<LinearGaussianProcessModel<State_, Input_>>::ProcessModelBase,
    public Traits<LinearGaussianProcessModel<State_, Input_>>::GaussianBase
{
private:
    /** Typdef of \c This for #from_traits(TypeName) helper */
    typedef LinearGaussianProcessModel<State_, Input_> This;

public:
    typedef from_traits(State);
    typedef from_traits(Input);
    typedef from_traits(Noise);
    typedef from_traits(Scalar);
    typedef from_traits(SecondMoment);
    typedef from_traits(DynamicsMatrix);

    typedef from_traits(GaussianBase);
    using GaussianBase::mean;
    using GaussianBase::covariance;
    using GaussianBase::dimension;
    using GaussianBase::square_root;

public:
    explicit
    LinearGaussianProcessModel(const SecondMoment& noise_covariance,
                               int dim = DimensionOf<State>()):
        Traits<This>::GaussianBase(dim),
        A_(DynamicsMatrix::Identity(dim, dim)),
        delta_time_(1.)
    {
        assert(dim > 0);

        covariance(noise_covariance);
    }


    /// \todo: this model has to be fixed. i needs to implement the solution
    /// to a linear differential equation
    explicit
    LinearGaussianProcessModel(int dim = DimensionOf<State>())
        : Traits<This>::GaussianBase(dim),
          A_(DynamicsMatrix::Identity(dim, dim)),
          delta_time_(1.)
    {
        assert(dim > 0);

        covariance(SecondMoment::Identity(dim, dim));
    }

    ~LinearGaussianProcessModel() { }

    virtual State predict_state(double delta_time,
                                const State& state,
                                const Noise& noise,
                                const Input& input)
    {
        condition(delta_time, state, input);

        return map_standard_normal(noise);
    }

    virtual int state_dimension() const
    {
        return Traits<This>::GaussianBase::dimension();
    }

    virtual int noise_dimension() const
    {
        return Traits<This>::GaussianBase::dimension();
    }

    virtual int input_dimension() const
    {
        return DimensionOf<Input>();
    }

    /// \todo: can we get rid of this function?
    virtual void condition(const double& delta_time,
                           const State& x,
                           const Input& u = Input())
    {
        delta_time_ = delta_time;

        mean(A_ * x);
    }

    virtual State map_standard_normal(const Noise& sample) const
    {
//        return mean() + delta_time_ * square_root() * sample; // TODO FIX TIME
        return mean() + square_root() * sample;
    }

    virtual const DynamicsMatrix& A() const
    {
        return A_;
    }

    virtual void A(const DynamicsMatrix& dynamics_matrix)
    {
        A_ = dynamics_matrix;
    }

protected:
    DynamicsMatrix A_;
    double delta_time_;
};

}

#endif
