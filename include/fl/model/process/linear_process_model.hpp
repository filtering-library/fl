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
#include <ff/models/process_models/interfaces/stationary_process_model.hpp>

namespace fl
{

// Forward declarations
template <typename State, typename Input_> class LinearGaussianProcessModel;

template <typename State_, typename Input_>
struct Traits<LinearGaussianProcessModel<State_, Input_>>
{
    typedef Gaussian<State_> GaussianBase;
    typedef StationaryProcessModel<State_, Input_> ProcessModelBase;

    typedef State_ State;
    typedef Input_ Input;
    typedef typename Traits<GaussianBase>::Scalar Scalar;
    typedef typename Traits<GaussianBase>::Operator Operator;
    typedef typename Traits<GaussianBase>::Noise Noise;

    typedef Eigen::Matrix<Scalar,
                          State::SizeAtCompileTime,
                          State::SizeAtCompileTime> DynamicsMatrix;
};

template <typename State_, typename Input_ = internal::Empty>
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
    typedef typename Traits<This>::Operator Operator;
    typedef typename Traits<This>::DynamicsMatrix DynamicsMatrix;

    using Traits<This>::GaussianBase::Mean;
    using Traits<This>::GaussianBase::Covariance;
    using Traits<This>::GaussianBase::Dimension;

public:
    LinearGaussianProcessModel(
            const Operator& noise_covariance,
            const size_t dimension = DimensionOf<State>()):
        Traits<This>::GaussianBase(dimension),
        A_(DynamicsMatrix::Identity(dimension, dimension)),
        delta_time_(1.)
    {
        Covariance(noise_covariance);
    }

    ~LinearGaussianProcessModel() { }

    virtual State MapStandardGaussian(const Noise& sample) const
    {
        return Mean() + delta_time_ * this->SquareRoot() * sample;
    }

    virtual void Condition(const double& delta_time,
                           const State& x,
                           const Input& u = Input())
    {
        delta_time_ = delta_time;                

        Mean(A_ * x);
    }

    virtual const DynamicsMatrix& A() const
    {
        return A_;
    }

    virtual void A(const DynamicsMatrix& dynamics_matrix)
    {
        A_ = dynamics_matrix;
    }

    virtual size_t InputDimension() const
    {
        return 0;
    }

protected:
    DynamicsMatrix A_;
    double delta_time_;    
};

}

#endif
