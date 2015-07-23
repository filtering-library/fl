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
#include <fl/util/descriptor.hpp>
#include <fl/distribution/gaussian.hpp>
#include <fl/distribution/decorrelated_gaussian.hpp>
#include <fl/model/adaptive_model.hpp>
#include <fl/model/observation/interface/observation_density.hpp>
#include <fl/model/observation/interface/observation_model_interface.hpp>
#include <fl/model/observation/interface/additive_observation_function.hpp>
#include <fl/model/observation/interface/additive_uncorrelated_observation_function.hpp>

namespace fl
{

/**
 * \ingroup observation_models
 */
template <typename Obsrv_, typename State_, typename Density>
class LinearObservationModel
    : public ObservationDensity<Obsrv_, State_>,                  // p(y|x)
      public AdditiveObservationFunction<Obsrv_, State_, Obsrv_> // H*x + N*v
{
public:
    typedef State_ State;
    typedef Obsrv_ Obsrv;

    typedef ObservationDensity<Obsrv, State> DensityInterface;
    typedef AdditiveObservationFunction<Obsrv, State, Obsrv> AdditiveInterface;
    typedef typename AdditiveInterface::FunctionInterface FunctionInterface;

    /**
     * Linear model density. The density for linear model is the Gaussian over
     * observation space.
     */
    //typedef Gaussian<Obsrv> Density;

    /**
     * Observation model sensor matrix \f$H_t\f$ use in
     *
     * \f$ y_t  = H_t x_t + N_t v_t \f$
     */
    typedef Eigen::Matrix<
                typename State::Scalar,
                SizeOf<Obsrv>::Value,
                SizeOf<State>::Value
            > SensorMatrix;

    /**
     * Observation model noise matrix \f$N_t\f$ use in
     *
     * \f$ y_t  = H_t x_t + N_t v_t\f$
     *
     * In this linear model, the noise model matrix is equivalent to the
     * square root of the covariance of the model density. Hence, the
     * NoiseMatrix type is the same type as the second moment of the density.
     *
     * This is equivalent to
     *  > typedef typename Gaussian<Obsrv>::SecondMoment NoiseMatrix;
     */
    typedef typename AdditiveInterface::NoiseMatrix NoiseMatrix;

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
     * \brief Overridable default destructor
     */
    virtual ~LinearObservationModel() { }

    /**
     * \brief expected_observation
     * \param state
     * \return
     */
    Obsrv expected_observation(const State& state) const OVERRIDE
    {
        return sensor_matrix_ * state;
    }

    Real log_probability(const Obsrv& obsrv, const State& state) const
    {
        density_.mean(expected_observation(state));

        return density_.log_probability(obsrv);
    }

    const SensorMatrix& sensor_matrix() const OVERRIDE
    {
        return sensor_matrix_;
    }

    const NoiseMatrix& noise_matrix() const OVERRIDE
    {
        return density_.square_root();
    }

    const NoiseMatrix& noise_covariance() const OVERRIDE
    {
        return density_.covariance();
    }

    int obsrv_dimension() const OVERRIDE
    {
        return sensor_matrix_.rows();
    }

    int noise_dimension() const OVERRIDE
    {
        return density_.square_root().cols();
    }

    int state_dimension() const OVERRIDE
    {
        return sensor_matrix_.cols();
    }

    virtual void sensor_matrix(const SensorMatrix& sensor_mat)
    {
        sensor_matrix_ = sensor_mat;
    }

    virtual void noise_matrix(const NoiseMatrix& noise_mat)
    {
        density_.square_root(noise_mat);
    }

    virtual void noise_covariance(const NoiseMatrix& noise_mat_squared)
    {
        density_.covariance(noise_mat_squared);
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
