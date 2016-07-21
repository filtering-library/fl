/*
 * This is part of the fl library, a C++ Bayesian filtering library
 * (https://github.com/filtering-library)
 *
 * Copyright (c) 2015 Max Planck Society,
 * 				 Autonomous Motion Department,
 * 			     Institute for Intelligent Systems
 *
 * This Source Code Form is subject to the terms of the MIT License (MIT).
 * A copy of the license can be found in the LICENSE file distributed with this
 * source code.
 */

/**
 * \file linear_decorrelated_gaussian_sensor.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once


#include <fl/util/traits.hpp>
#include <fl/util/types.hpp>
#include <fl/util/descriptor.hpp>
#include <fl/distribution/decorrelated_gaussian.hpp>
#include <fl/model/sensor/linear_sensor.hpp>
#include <fl/model/sensor/interface/additive_uncorrelated_sensor_function.hpp>

namespace fl
{

/**
 * \ingroup sensors
 */
template <typename Obsrv, typename State>
class LinearDecorrelatedGaussianSensor
    : public AdditiveUncorrelatedNoiseModel<DecorrelatedGaussian<Obsrv>>,
      public AdditiveSensorFunction<Obsrv, State, Gaussian<Obsrv>>,
      public SensorDensity<Obsrv, State>,
      public Descriptor,
      private internal::LinearModelType
{
public:
    // specify the main type of this model
    typedef internal::AdditiveUncorrelatedNoiseModelType Type;

    typedef AdditiveUncorrelatedNoiseModel<
                DecorrelatedGaussian<Obsrv>
            > AdditiveUncorrelatedInterface;

    typedef AdditiveSensorFunction<
                Obsrv, State, Gaussian<Obsrv>
            > AdditiveSensorFunctionInterface;

    typedef
    typename AdditiveSensorFunctionInterface::NoiseMatrix NoiseMatrix;

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
    LinearDecorrelatedGaussianSensor(
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
    virtual ~LinearDecorrelatedGaussianSensor() noexcept { }

    /**
     * \brief expected_observation
     * \param state
     * \return
     */
    Obsrv expected_observation(const State& state) const override
    {
        return sensor_matrix_ * state;
    }

    Real log_probability(const Obsrv& obsrv, const State& state) const override
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

    virtual NoiseDiagonalMatrix noise_diagonal_matrix() const
    {
        return density_.square_root();
    }

    virtual NoiseDiagonalMatrix noise_diagonal_covariance() const
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

    virtual void noise_diagonal_matrix(
        const NoiseDiagonalMatrix& noise_mat)
    {
        density_.square_root(noise_mat);
    }

    virtual void noise_diagonal_covariance(
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

    virtual NoiseMatrix create_noise_covariance() const
    {
        auto C = noise_covariance();
        C.setIdentity();
        return C;
    }

    virtual NoiseMatrix create_noise_diagonal_matrix() const
    {
        auto N = noise_diagonal_matrix();
        N.setIdentity();
        return N;
    }

    virtual NoiseMatrix create_noise_diagonal_covariance() const
    {
        auto C = noise_diagonal_covariance();
        C.setIdentity();
        return C;
    }

    virtual std::string name() const
    {
        return "LinearDecorrelatedGaussianSensor";
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


