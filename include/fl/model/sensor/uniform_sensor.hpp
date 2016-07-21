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
 * \file uniform_sensor.hpp
 * \date September 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

#include <Eigen/Dense>

#include <fl/util/descriptor.hpp>
#include <fl/distribution/uniform_distribution.hpp>
#include <fl/model/sensor/interface/sensor_density.hpp>
#include <fl/model/sensor/interface/sensor_function.hpp>

namespace fl
{

template <typename State_>
class UniformSensor
    : public SensorFunction<Vector1d, State_, Vector1d>,
      public SensorDensity<Vector1d, State_>,
      public Descriptor
{
public:
    typedef Vector1d Obsrv;
    typedef Vector1d Noise;
    typedef State_   State;

public:
    UniformSensor(
        Real min_value,
        Real max_value,
        int state_dim = DimensionOf<State>::Value)
        : state_dim_(state_dim),
          density_(min_value, max_value)
    { }

    Real log_probability(const Obsrv& obsrv, const State& state) const override
    {
        return density_.log_probability(obsrv);
    }

    Real probability(const Obsrv& obsrv, const State& state) const override
    {
        return density_.probability(obsrv);
    }

    Obsrv observation(const State& state, const Noise& noise) const override
    {
        Obsrv y = density_.map_standard_normal(noise);
        return y;
    }

    virtual int obsrv_dimension() const { return 1; }
    virtual int noise_dimension() const { return 1; }
    virtual int state_dimension() const { return state_dim_; }

    virtual std::string name() const
    {
        return "UniformSensor";
    }

    virtual std::string description() const
    {
        return "UniformSensor";
    }

private:
    int state_dim_;
    UniformDistribution density_;
};

}
