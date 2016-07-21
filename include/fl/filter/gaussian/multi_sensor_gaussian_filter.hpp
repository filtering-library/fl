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
 * \file multi_sensor_gaussian_filter.hpp
 * \date August 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

#include <Eigen/Dense>

#include <fl/util/meta.hpp>
#include <fl/util/traits.hpp>
#include <fl/filter/gaussian/gaussian_filter.hpp>
#include <fl/filter/gaussian/update_policy/multi_sensor_sigma_point_update_policy.hpp>
#include <fl/filter/gaussian/update_policy/multi_sensor_sigma_point_update_polizzle.hpp>

namespace fl
{
// Forward declarations
template <typename...>
class MultiSensorGaussianFilter;

/**
 * \internal
 * \ingroup nonlinear_gaussian_filter
 *
 * Traits of GaussianFilter for multi sensors based on quadrature with
 * customizable policies, i.e implementations of the time and measurement
 * updates.
 */
template <typename TransitionFunction,
          typename JointSensorFunction,
          typename... Policies>
struct Traits<MultiSensorGaussianFilter<TransitionFunction,
                                        JointSensorFunction,
                                        Policies...>>
{
    typedef typename TransitionFunction::State State;
    typedef typename TransitionFunction::Input Input;
    typedef typename JointSensorFunction::Obsrv Obsrv;
    typedef Gaussian<State> Belief;
};

/**
 * \ingroup nonlinear_gaussian_filter
 */
template <typename TransitionFunction,
          typename JointSensorFunction,
          typename Quadrature>
class MultiSensorGaussianFilter<TransitionFunction,
                                JointSensorFunction,
                                Quadrature> :
/* Implement the filter interface */
#ifndef GENERATING_DOCUMENTATION
    public GaussianFilter<
        typename RemoveAdditivityOf<TransitionFunction>::Type,
        typename RemoveAdditivityOf<JointSensorFunction>::Type,
        Quadrature,
        SigmaPointPredictPolicy<
            Quadrature,
            typename AdditivityOf<TransitionFunction>::Type>,
        MultiSensorSigmaPointUpdatePolizzle<
//        MultiSensorSigmaPointUpdatePolicy<
            Quadrature,
            typename AdditivityOf<JointSensorFunction>::Type>>
#else
    public GaussianFilter<TransitionFunction,
                          SensorFunction,
                          Quadrature,
                          PredictionPolicy,
                          MultiSensorUpdatePolicy>
#endif
{
public:
    typedef GaussianFilter<
        typename RemoveAdditivityOf<TransitionFunction>::Type,
        typename RemoveAdditivityOf<JointSensorFunction>::Type,
        Quadrature,
        SigmaPointPredictPolicy<
            Quadrature,
            typename AdditivityOf<TransitionFunction>::Type>,
        MultiSensorSigmaPointUpdatePolizzle<
//        MultiSensorSigmaPointUpdatePolicy<
            Quadrature,
            typename AdditivityOf<JointSensorFunction>::Type>> Base;

    MultiSensorGaussianFilter(const TransitionFunction& transition,
                              const JointSensorFunction& sensor,
                              const Quadrature& quadrature)
        : Base(transition, sensor, quadrature)
    {
    }

    /**
     * \brief Overridable default destructor
     */
    virtual ~MultiSensorGaussianFilter() noexcept {}
    virtual std::string name() const
    {
        return "MultiSensorGaussianFilter<" +
               this->list_arguments(Base::name()) + ">";
    }

    virtual std::string description() const
    {
        return "Multi-Sensor Gaussian Filter with" +
               this->list_descriptions(Base::description());
    }
};
}
