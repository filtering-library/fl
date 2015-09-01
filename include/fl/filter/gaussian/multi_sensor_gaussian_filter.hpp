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
 * \file multi_sensor_gaussian_filter.hpp
 * \date August 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__FILTER__GAUSSIAN__MULTI_SENSOR_GAUSSIAN_FILTER_HPP
#define FL__FILTER__GAUSSIAN__MULTI_SENSOR_GAUSSIAN_FILTER_HPP


#include <Eigen/Dense>

#include <fl/util/meta.hpp>
#include <fl/util/traits.hpp>
#include <fl/filter/gaussian/gaussian_filter.hpp>
#include <fl/filter/gaussian/multi_sensor_sigma_point_update_policy.hpp>

namespace fl
{

// Forward declarations
template <typename...> class MultiSensorGaussianFilter;

/**
 * \internal
 * \ingroup nonlinear_gaussian_filter
 *
 * Traits of GaussianFilter for multi sensors based on quadrature with
 * customizable policies, i.e implementations of the time and measurement
 * updates.
 */
template <
    typename StateTransitionFunction,
    typename JointObservationFunction,
    typename ... Policies
>
struct Traits<
          MultiSensorGaussianFilter<
              StateTransitionFunction, JointObservationFunction, Policies...>>
{
    typedef typename StateTransitionFunction::State State;
    typedef typename StateTransitionFunction::Input Input;
    typedef typename JointObservationFunction::Obsrv Obsrv;
    typedef Gaussian<State> Belief;
};


/**
 * \ingroup nonlinear_gaussian_filter
 */
template<
    typename StateTransitionFunction,
    typename JointObservationFunction,
    typename Quadrature
>
class MultiSensorGaussianFilter<
          StateTransitionFunction,
          JointObservationFunction,
          Quadrature>
    :
    /* Implement the filter interface */
#ifndef GENERATING_DOCUMENTATION
    public GaussianFilter<
               typename RemoveAdditivityOf<StateTransitionFunction>::Type,
               typename RemoveAdditivityOf<JointObservationFunction>::Type,
               Quadrature,
               SigmaPointPredictPolicy<
                   Quadrature,
                   typename AdditivityOf<StateTransitionFunction>::Type>,
               MultiSensorSigmaPointUpdatePolicy<
                   Quadrature,
                   typename AdditivityOf<JointObservationFunction>::Type>>
#else
    public GaussianFilter<
               StateTransitionFunction,
               ObservationFunction,
               Quadrature,
               PredictionPolicy,
               MultiSensorUpdatePolicy>
#endif
{
public:
    typedef GaussianFilter<
                typename RemoveAdditivityOf<StateTransitionFunction>::Type,
                typename RemoveAdditivityOf<JointObservationFunction>::Type,
                Quadrature,
                SigmaPointPredictPolicy<
                    Quadrature,
                    typename AdditivityOf<StateTransitionFunction>::Type>,
                MultiSensorSigmaPointUpdatePolicy<
                    Quadrature,
                    typename AdditivityOf<JointObservationFunction>::Type>
            > Base;

    MultiSensorGaussianFilter(
        const StateTransitionFunction& process_model,
        const JointObservationFunction& obsrv_model,
        const Quadrature& quadrature)
        : Base(process_model, obsrv_model, quadrature)
    { }

    /**
     * \brief Overridable default destructor
     */
    virtual ~MultiSensorGaussianFilter() { }

    virtual std::string name() const
    {
        return "MultiSensorGaussianFilter<"
                + this->list_arguments(Base::name())
                + ">";
    }

    virtual std::string description() const
    {
        return "Multi-Sensor Gaussian Filter with"
                + this->list_descriptions(Base::description());
    }
};

}

#endif
