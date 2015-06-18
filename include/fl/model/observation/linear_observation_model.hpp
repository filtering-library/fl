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
#include <fl/model/adaptive_model.hpp>
#include <fl/model/observation/interface/observation_density.hpp>
#include <fl/model/observation/interface/observation_model_interface.hpp>
#include <fl/model/observation/interface/additive_observation_function.hpp>

namespace fl
{

// Forward declarations
template <
    typename Obsrv,
    typename State
>
class LinearGaussianObservationModel;


/**
 * Linear Gaussian observation model traits. This trait definition contains all
 * types used internally within the model. Additionally, it provides the types
 * needed externally to use the linear Gaussian model.
 */
template <
    typename Obsrv_,
    typename State_
>
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

    typedef Eigen::Matrix<Scalar, 1, 1> Param;
    typedef AdaptiveModel<Param> AdaptiveModelBase;

    typedef Eigen::Matrix<
                Scalar,
                Obsrv::SizeAtCompileTime,
                Param::SizeAtCompileTime
            > ParamMatrix;
};

/**
 * \ingroup observation_models
 */
template <typename Obsrv,typename State>
class LinearGaussianObservationModel
    : public Traits<
                 LinearGaussianObservationModel<Obsrv, State>
             >::GaussianBase,
      public Traits<
                 LinearGaussianObservationModel<Obsrv, State>
             >::ObservationModelBase,
      public Traits<
                 LinearGaussianObservationModel<Obsrv, State>
             >::AdaptiveModelBase
{
private:
    /** Typdef of \c This for #from_traits(TypeName) helper */
    typedef LinearGaussianObservationModel<Obsrv, State> This;

public:
    typedef from_traits(Scalar);
    typedef from_traits(Noise);
    typedef from_traits(Param);
    typedef from_traits(SecondMoment);
    typedef from_traits(SensorMatrix);
    typedef from_traits(ParamMatrix);

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
          /// \todo: different initialization. ideally something which
          /// corresponds to identity when state and observation dimensions
          /// are identical.
          H_(SensorMatrix::Ones(obsrv_dimension(),
                                state_dimension())),
          H_p_(ParamMatrix::Ones(obsrv_dimension(),1)),
          param_(Param::Zero(1, 1))
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
                                state_dimension())),
          H_p_(ParamMatrix::Ones(obsrv_dimension(),1)),
          param_(Param::Zero())
    {
        covariance(SecondMoment::Identity(obsrv_dim, obsrv_dim));
    }

    ~LinearGaussianObservationModel() { }

    virtual void condition(const State& x)
    {
        mean(H_ * x + H_p_ * param_);
    }

    virtual const SensorMatrix& H() const
    {
        return H_;
    }

    virtual void H(const SensorMatrix& sensor_matrix)
    {
        H_ = sensor_matrix;
    }

    /// \todo: get rid of delta_time
    virtual Obsrv predict_obsrv(const State& state,
                                      const Noise& noise,
                                      double delta_time)
    {
        condition(state);
        Obsrv y = Traits<This>::GaussianBase::map_standard_normal(noise);
        return y;
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

    virtual const Param& param() const
    {
        return param_;
    }

    virtual void param(Param params)
    {
        param_ = params;
    }

    virtual int param_dimension() const
    {
        return param_.rows();
    }

protected:
    int state_dimension_;
    SensorMatrix H_;

    /// \todo: get rid of param_ and H_p_
    ParamMatrix H_p_;
    Param param_;
};


/**
 * \ingroup observation_models
 *
 * This represents the linear gaussian observation model \f$p(y_t \mid x_t)\f$
 * governed by the linear equation
 *
 * \f$ y_t  = H_t x_t + N_t v_t \f$
 *
 * where \f$ y_t\f$ is the observation, \f$ x_t\f$ is the state. The matrix
 * \f$H_t\f$ is the sensor model mapping the state into the observation space.
 * The vector \f$ v_t \sim {\cal N}(v_t; 0, I)\f$ is a standard normal variate
 * which is mapped into the the observation space via the noise model matrix
 * \f$N_t\f$. Any Gaussian noise \f$\tilde{v}_t \sim {\cal N}(\tilde{v}_t ; 0,
 * R_t) \f$ can be represented via \f$\tilde{v}_t = N_t v_t\f$ with\f$N_t
 * = \sqrt{R_t}\f$. Hence, the linear equation may be restated as the more
 * familiar but equivalent form
 *
 * \f$ y_t  = H_t x_t + \tilde{v}_t \f$.
 */
template <typename Obsrv_, typename State_>
class LinearObservationModel
    : public ObservationDensity<Obsrv_, State_>,                  // p(y|x)
      public AdditiveObservationFunction<Obsrv_, State_, Obsrv_>  // H*x + N*v
{
public:
    typedef Obsrv_ Obsrv;
    typedef State_ State;

    /**
     * Linear model density. The density for linear model is the Gaussian over
     * observation space.
     */
    typedef Gaussian<Obsrv> Density;

    /**
     * Observation model sensor matrix \f$H_t\f$ use in
     *
     * \f$ y_t  = H_t x_t + N_t v_t \f$
     */
    typedef Eigen::Matrix<
                typename Obsrv::Scalar,
                Obsrv::SizeAtCompileTime,
                State::SizeAtCompileTime
            > SensorMatrix;

    /**
     * Observation model noise matrix \f$N_t\f$ use in
     *
     * \f$ y_t  = H_t x_t + N_t v_t\f$
     *
     * In this linear model, the noise model matrix is equivalent to the
     * square root of the covariance of the model density. Hence, the
     * NoiseMatrix type is the same type as the second moment of the density.
     */
    typedef typename Density::SecondMoment NoiseMatrix;

public:
    /**
     * Constructs a linear gaussian observation model
     * \param obsrv_dim     observation dimension if dynamic size
     * \param state_dim     state dimension if dynamic size
     */
    explicit
    LinearObservationModel(int obsrv_dim = DimensionOf<Obsrv>(),
                           int state_dim = DimensionOf<State>())
        : sensor_matrix_(SensorMatrix::Identity(obsrv_dim, state_dim)),
          density_(obsrv_dim)

    {
        assert(obsrv_dim > 0);
        assert(state_dim > 0);
    }

    /**
     * \brief Overridable default destructor     */
    virtual ~LinearObservationModel() { }

    /**
     * \brief expected_observation
     * \param state
     * \return
     */
    virtual Obsrv expected_observation(const State& state) const
    {
        return sensor_matrix_ * state;
    }

    virtual FloatingPoint log_probability(const Obsrv& obsrv,
                                          const State& state) const
    {
        density_.mean(expected_observation(state));

        return density_.log_probability(obsrv);
    }

    virtual const SensorMatrix& sensor_matrix() const
    {
        return sensor_matrix_;
    }

    virtual void sensor_matrix(const SensorMatrix& sensor_mat)
    {
        sensor_matrix_ = sensor_mat;
    }

    virtual const NoiseMatrix& noise_matrix() const
    {
        return density_.square_root();
    }

    virtual void noise_matrix(const NoiseMatrix& noise_mat)
    {
        density_.square_root(noise_mat);
    }

    virtual NoiseMatrix noise_matrix_squared() const
    {
        return density_.covariance();
    }

    virtual void noise_matrix_squared(const NoiseMatrix& noise_mat_squared)
    {
        density_.covariance(noise_mat_squared);
    }

    virtual int obsrv_dimension() const
    {
        return sensor_matrix_.rows();
    }

    virtual int noise_dimension() const
    {
        return density_.square_root().cols();
    }

    virtual int state_dimension() const
    {
        return sensor_matrix_.cols();
    }

    virtual SensorMatrix create_sensor_matrix() const
    {
        auto H = sensor_matrix();
        H.setIdentity();
        return H;
    }

    virtual NoiseMatrix create_noise_matrix() const
    {
        auto N = noise_matrix();
        N.setIdentity();
        return N;
    }

protected:
    SensorMatrix sensor_matrix_;
    mutable Density density_;
};

}

#endif
