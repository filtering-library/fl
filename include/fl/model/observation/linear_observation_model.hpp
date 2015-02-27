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
 * \file linear_observation_model.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__MODEL__OBSERVATION__LINEAR_OBSERVATION_MODEL_HPP
#define FL__MODEL__OBSERVATION__LINEAR_OBSERVATION_MODEL_HPP

#include <fl/util/traits.hpp>
#include <fl/distribution/gaussian.hpp>
#include <fl/model/observation/observation_model_interface.hpp>

namespace fl
{

// Forward declarations
template <
    typename Obsrv,
    typename State>
class LinearGaussianObservationModel;


/**
 * Linear Gaussian observation model traits. This trait definition contains all
 * types used internally within the model. Additionally, it provides the types
 * needed externally to use the linear Gaussian model.
 */
template <
    typename Obsrv_,
    typename State_>
struct Traits<
           LinearGaussianObservationModel<Obsrv_, State_>>
{    
    typedef State_ State;
    typedef Obsrv_ Obsrv;

    typedef Gaussian<Obsrv> GaussianBase;
    typedef typename Traits<GaussianBase>::Scalar Scalar;
    typedef typename Traits<GaussianBase>::SecondMoment SecondMoment;
    typedef typename Traits<GaussianBase>::StandardVariate Noise;

    typedef Eigen::Matrix<
                Scalar,
                Obsrv::SizeAtCompileTime,
                State::SizeAtCompileTime
            > SensorMatrix;

    typedef ObservationModelInterface<
                Obsrv,
                State,
                Noise
            > ObservationModelBase;
};

/**
 * \ingroup observation_models
 */
template <typename Obsrv,typename State>
class LinearGaussianObservationModel
    : public Traits<
                 LinearGaussianObservationModel<Obsrv, State>
             >::ObservationModelBase,
      public Traits<
                 LinearGaussianObservationModel<Obsrv, State>
             >::GaussianBase
{
public:
    typedef LinearGaussianObservationModel<Obsrv, State> This;

    typedef typename Traits<This>::Noise Noise;
    typedef typename Traits<This>::Scalar Scalar;
    typedef typename Traits<This>::SecondMoment SecondMoment;
    typedef typename Traits<This>::SensorMatrix SensorMatrix;

    using Traits<This>::GaussianBase::mean;
    using Traits<This>::GaussianBase::covariance;
    using Traits<This>::GaussianBase::dimension;

public:
    explicit
    LinearGaussianObservationModel(
            const SecondMoment& noise_covariance,
            const int obsrv_dim = DimensionOf<Obsrv>(),
            const int state_dim = DimensionOf<State>())
        : Traits<This>::GaussianBase(obsrv_dim),
          state_dimension_(state_dim),
          H_(SensorMatrix::Ones(obsrv_dimension(),
                                state_dimension()))
    {
        covariance(noise_covariance);
    }

    explicit
    LinearGaussianObservationModel(
            const int obsrv_dim = DimensionOf<Obsrv>(),
            const int state_dim = DimensionOf<State>())
        : Traits<This>::GaussianBase(obsrv_dim),
          state_dimension_(state_dim),
          H_(SensorMatrix::Ones(obsrv_dimension(),
                                state_dimension()))
    {
        covariance(SecondMoment::Identity(obsrv_dim, obsrv_dim));
    }

    ~LinearGaussianObservationModel() { }

    virtual void condition(const State& x)
    {
        mean(H_ * x);
    }

    virtual const SensorMatrix& H() const
    {
        return H_;
    }

    virtual void H(const SensorMatrix& sensor_matrix)
    {
        H_ = sensor_matrix;
    }

    virtual Obsrv predict_obsrv(const State& state,
                                      const Noise& noise,
                                      double delta_time)
    {
        condition(state);
        return Traits<This>::GaussianBase::map_standard_normal(noise);
    }

    virtual int obsrv_dimension() const
    {
        return Traits<This>::GaussianBase::dimension();
    }

    virtual int noise_dimension() const
    {
        return Traits<This>::GaussianBase::standard_variate_dimension();
    }

    virtual int state_dimension() const
    {
        return state_dimension_;
    }


protected:
    int state_dimension_;
    SensorMatrix H_;
};

}

#endif

