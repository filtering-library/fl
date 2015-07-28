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
 * \file linear_decorrelated_gaussian_observation_model.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__MODEL__OBSERVATION__LINEAR_DECORRELATED_GAUSSIAN_OBSERVATION_MODEL_HPP
#define FL__MODEL__OBSERVATION__LINEAR_DECORRELATED_GAUSSIAN_OBSERVATION_MODEL_HPP

#include <fl/util/traits.hpp>
#include <fl/util/types.hpp>
#include <fl/util/descriptor.hpp>
#include <fl/distribution/decorrelated_gaussian.hpp>
#include <fl/model/observation/linear_observation_model.hpp>
#include <fl/model/observation/interface/additive_uncorrelated_observation_function.hpp>

namespace fl
{

/**
 * \ingroup observation_models
 */
template <typename Obsrv, typename State>
class LinearDecorrelatedGaussianObservationModel
    : public AdditiveUncorrelatedNoiseModel<DecorrelatedGaussian<Obsrv>>,
      public AdditiveObservationFunction<Obsrv, State, Gaussian<Obsrv>>,
      public Descriptor,
      private internal::LinearModelType
{
public:
    // specify the main type of this model
    typedef internal::AdditiveUncorrelatedNoiseModelType Type;

    typedef AdditiveUncorrelatedNoiseModel<
                DecorrelatedGaussian<Obsrv>
            > AdditiveUncorrelatedInterface;

    typedef AdditiveObservationFunction<
                Obsrv, State, Gaussian<Obsrv>
            > AdditiveObservationFunctionInterface;

    typedef
    typename AdditiveObservationFunctionInterface::NoiseMatrix NoiseMatrix;

    typedef
    typename AdditiveUncorrelatedInterface::NoiseMatrix NoiseDiagonalMatrix;

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
     * Constructs a linear gaussian observation model
     *
     * \param obsrv_dim     observation dimension if dynamic size
     * \param state_dim     state dimension if dynamic size
     */
    explicit
    LinearDecorrelatedGaussianObservationModel(
        int obsrv_dim = DimensionOf<Obsrv>(),
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
    virtual ~LinearDecorrelatedGaussianObservationModel() { }

    virtual NoiseDiagonalMatrix noise_matrix_diagonal() const
    {
        return density_.square_root();
    }

    virtual NoiseDiagonalMatrix noise_covariance_diagonal() const
    {
        return density_.covariance();
    }

    /**
     * \brief expected_observation
     * \param state
     * \return
     */
    Obsrv expected_observation(const State& state) const override
    {
        return sensor_matrix_ * state;
    }

    Real log_probability(const Obsrv& obsrv, const State& state) const
    {
        density_.mean(expected_observation(state));

        return density_.log_probability(obsrv);
    }

    const SensorMatrix& sensor_matrix() const override
    {
        return sensor_matrix_;
    }

    NoiseMatrix noise_matrix() const override
    {
        return density_.square_root();
    }

    NoiseMatrix noise_covariance() const override
    {
        return density_.covariance();
    }

    int obsrv_dimension() const override
    {
        return sensor_matrix_.rows();
    }

    int noise_dimension() const override
    {
        return density_.square_root().cols();
    }

    int state_dimension() const override
    {
        return sensor_matrix_.cols();
    }

    virtual void sensor_matrix(const SensorMatrix& sensor_mat)
    {
        sensor_matrix_ = sensor_mat;
    }

    virtual void noise_matrix(const NoiseMatrix& noise_mat)
    {
        density_.square_root(noise_mat.diagonal().asDiagonal());
    }

    virtual void noise_covariance(const NoiseMatrix& noise_mat_squared)
    {
        density_.covariance(noise_mat_squared.diagonal().asDiagonal());
    }

    virtual void noise_matrix_diagonal(
        const NoiseDiagonalMatrix& noise_mat)
    {
        density_.square_root(noise_mat);
    }

    virtual void noise_covariance_diagonal(
        const NoiseDiagonalMatrix& noise_mat_squared)
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

    virtual std::string name() const
    {
        return "LinearDecorrelatedGaussianObservationModel";
    }

    virtual std::string description() const
    {
        return "Linear observation model with additive decorrelated Gaussian "
               "noise";
    }

private:
    SensorMatrix sensor_matrix_;
    mutable DecorrelatedGaussian<Obsrv> density_;
};

}

#endif
