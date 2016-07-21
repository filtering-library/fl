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
 * \file linear_cauchy_sensor.hpp
 * \date August 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once


#include <fl/util/traits.hpp>
#include <fl/util/types.hpp>
#include <fl/util/descriptor.hpp>
#include <fl/distribution/cauchy_distribution.hpp>
#include <fl/model/sensor/interface/sensor_density.hpp>
#include <fl/model/sensor/interface/sensor_function.hpp>


namespace fl
{

/**
 * \ingroup sensors
 */
template <typename Obsrv, typename State>
class LinearCauchySensor
    : public SensorFunction<
                Obsrv,
                State,
                typename CauchyDistribution<Obsrv>::StandardVariate>, // noise
      public SensorDensity<Obsrv, State>,
      public Descriptor
{
public:
    typedef typename CauchyDistribution<Obsrv>::StandardVariate Noise;
    typedef typename CauchyDistribution<Obsrv>::SecondMoment NoiseMatrix;

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
    LinearCauchySensor(
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
    virtual ~LinearCauchySensor() noexcept { }

    Obsrv observation(const State& state, const Noise& noise) const override
    {
        return sensor_matrix_ * state + density_.map_standard_normal(noise);
    }

    Real log_probability(const Obsrv& obsrv, const State& state) const override
    {
        density_.location(sensor_matrix_ * state);

        return density_.log_probability(obsrv);
    }

    const SensorMatrix& sensor_matrix() const override
    {
        return sensor_matrix_;
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
        return density_.standard_variate_dimension();
    }

    int state_dimension() const override
    {
        return sensor_matrix_.cols();
    }

    virtual void sensor_matrix(const SensorMatrix& sensor_mat)
    {
        sensor_matrix_ = sensor_mat;
    }

    virtual void noise_covariance(const NoiseMatrix& noise_mat)
    {
        density_.scaling_matrix(noise_mat);
    }

    virtual SensorMatrix create_sensor_matrix() const
    {
        auto H = sensor_matrix();
        H.setIdentity();
        return H;
    }

    virtual std::string name() const
    {
        return "LinearCauchySensor";
    }

    virtual std::string description() const
    {
        return "Linear cauchy observation model with Cauchy distribution noise";
    }

private:
    SensorMatrix sensor_matrix_;
    mutable CauchyDistribution<Obsrv> density_;
};

}


