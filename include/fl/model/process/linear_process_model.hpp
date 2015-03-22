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
#include <fl/model/process/process_model_interface.hpp>

namespace fl
{

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
public:
    typedef LinearGaussianProcessModel<State_, Input_> This;

    typedef typename Traits<This>::State State;
    typedef typename Traits<This>::Input Input;
    typedef typename Traits<This>::Noise Noise;
    typedef typename Traits<This>::Scalar Scalar;
    typedef typename Traits<This>::SecondMoment SecondMoment;
    typedef typename Traits<This>::DynamicsMatrix DynamicsMatrix;

    using Traits<This>::GaussianBase::mean;
    using Traits<This>::GaussianBase::covariance;
    using Traits<This>::GaussianBase::dimension;
    using Traits<This>::GaussianBase::square_root;

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
