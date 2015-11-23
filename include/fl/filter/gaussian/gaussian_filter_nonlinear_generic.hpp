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
 * \file gaussian_filter_nonlinear_generic.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once


#include <Eigen/Dense>

#include <fl/util/meta.hpp>
#include <fl/util/traits.hpp>
#include <fl/filter/filter_interface.hpp>
#include <fl/distribution/gaussian.hpp>

namespace fl
{

/**
* \defgroup generic_nonlinear_gaussian_filter Generic Nonlinear Gaussian Filter
* \ingroup filters
*/

// Forward delcaration
template <typename...> class GaussianFilter;

/**
 * \internal
 * \ingroup nonlinear_gaussian_filter
 * \ingroup generic_nonlinear_gaussian_filter
 *
 * Traits for generic GaussianFilter based on quadrature (numeric integration)
 * with customizable policies, i.e implementations of the time and measurement
 * updates.
 */
template <
    typename StateTransitionModel,
    typename ObservationModel,
    typename Quadrature,
    typename ... Policies
>
struct Traits<
           GaussianFilter<
               StateTransitionModel, ObservationModel, Quadrature, Policies...>>
{
    typedef typename StateTransitionModel::State State;
    typedef typename StateTransitionModel::Input Input;
    typedef typename ObservationModel::Obsrv Obsrv;
    typedef Gaussian<State> Belief;
};

/**
 * \ingroup nonlinear_gaussian_filter
 * \ingroup generic_nonlinear_gaussian_filter
 *
 * GaussianFilter represents all filters based on Gaussian distributed systems.
 * This includes the Kalman Filter and filters using non-linear models such as
 * Sigma Point Kalman Filter family.
 *
 * \tparam StateTransitionFunction
 * \tparam ObservationFunction
 * \tparam Quadrature
 * \tparam PredictionPolicy
 * \tparam UpdatePolicy
 */
template<
    typename StateTransitionFunction,
    typename ObservationFunction,
    typename Quadrature,
    typename PredictionPolicy,
    typename UpdatePolicy
>
class GaussianFilter<
          StateTransitionFunction,
          ObservationFunction,
          Quadrature,
          PredictionPolicy,
          UpdatePolicy>
    :
    /* Implement the filter interface */
    public FilterInterface<
               GaussianFilter<
                   StateTransitionFunction,
                   ObservationFunction,
                   Quadrature,
                   PredictionPolicy,
                   UpdatePolicy>>
{
public:
    typedef typename StateTransitionFunction::State State;
    typedef typename StateTransitionFunction::Input Input;
    typedef typename ObservationFunction::Obsrv Obsrv;
    typedef Gaussian<State> Belief;

public:
    /**
     * Creates a Gaussian filter
     *
     * \param process_model         Process model instance
     * \param obsrv_model           Obsrv model instance
     * \param transform   Point set tranfrom such as the unscented
     *                              transform
     */
    GaussianFilter(const StateTransitionFunction& process_model,
                   const ObservationFunction& obsrv_model,
                   const Quadrature& quadrature)
        : process_model_(process_model),
          obsrv_model_(obsrv_model),
          quadrature_(quadrature)
    { }

    /**
     * \brief Overridable default destructor
     */
    virtual ~GaussianFilter() { }

    /**
     * \copydoc FilterInterface::predict
     */
    virtual void predict(const Belief& prior_belief,
                         const Input& input,
                         Belief& predicted_belief)
    {
        prediction_policy_(process_model(),
                           quadrature(),
                           prior_belief,
                           input,
                           predicted_belief);
    }

    /**
     * \copydoc FilterInterface::update
     */
    virtual void update(const Belief& predicted_belief,
                        const Obsrv& obsrv,
                        Belief& posterior_belief)
    {
        update_policy_(obsrv_model(),
                       quadrature(),
                       predicted_belief,
                       obsrv,
                       posterior_belief);
    }

public: /* factory functions */
    virtual Belief create_belief() const
    {
        auto belief = Belief(process_model().state_dimension());
        return belief; // RVO
    }

public: /* accessors & mutators */
    StateTransitionFunction& process_model()
    {
        return process_model_;
    }

    ObservationFunction& obsrv_model()
    {
        return obsrv_model_;
    }

    Quadrature& quadrature()
    {
        return quadrature_;
    }

    const StateTransitionFunction& process_model() const
    {
        return process_model_;
    }

    const ObservationFunction& obsrv_model() const
    {
        return obsrv_model_;
    }

    const Quadrature& quadrature() const
    {
        return quadrature_;
    }

    virtual std::string name() const
    {
        return "GaussianFilter<"
                + this->list_arguments(
                            process_model().name(),
                            obsrv_model().name(),
                            quadrature().name(),
                            prediction_policy_.name(),
                            update_policy_.name())
                + ">";
    }

    virtual std::string description() const
    {
        return "Sigma point based GaussianFilter with"
                + this->list_descriptions(
                            process_model().description(),
                            obsrv_model().description(),
                            quadrature().description(),
                            prediction_policy_.description(),
                            update_policy_.description());
    }

protected:
    /** \cond internal */
    StateTransitionFunction process_model_;
    ObservationFunction obsrv_model_;
    Quadrature quadrature_;
    PredictionPolicy prediction_policy_;
    UpdatePolicy update_policy_;
    /** \endcond */
};

}


