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
 * \file robust_gaussian_filter.hpp
 * \date August 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__FILTER__GAUSSIAN__ROBUST_GAUSSIAN_FILTER_HPP
#define FL__FILTER__GAUSSIAN__ROBUST_GAUSSIAN_FILTER_HPP

#include <fl/util/meta.hpp>
#include <fl/util/traits.hpp>
#include <fl/filter/gaussian/gaussian_filter_nonlinear_generic.hpp>
#include <fl/filter/gaussian/sigma_point_additive_uncorrelated_update_policy.hpp>
#include <fl/filter/gaussian/sigma_point_additive_prediction_policy.hpp>
#include <fl/filter/gaussian/sigma_point_additive_update_policy.hpp>S
#include <fl/filter/gaussian/sigma_point_prediction_policy.hpp>
#include <fl/filter/gaussian/sigma_point_update_policy.hpp>

namespace fl
{


// Forward delcaration
template<
    typename StateTransitionFunction,
    typename ObservationFunction,
    typename ... Policies
> class RobustGaussianFilter;

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
    typename StateTransitionFunction,
    typename ObservationFunction,
    typename ... Policies
>
struct Traits<
          RobustGaussianFilter<
              StateTransitionFunction, ObservationFunction, Policies...>>
{
    typedef typename StateTransitionFunction::State State;
    typedef typename StateTransitionFunction::Input Input;
    typedef typename ObservationFunction::Obsrv Obsrv;
    typedef Gaussian<State> Belief;
};




//template <typename Obsrv, typename State, typename FeatureFunction>
//class FeatureAugmentedObservationModel
//{
//public:

//};




/**
 * \ingroup nonlinear_gaussian_filter
 * \ingroup generic_nonlinear_gaussian_filter
 */
template<
    typename StateTransitionFunction,
    typename ObservationFunction,
    typename ... Policies
>
class RobustGaussianFilter
    : public FilterInterface<
                 RobustGaussianFilter<
                     StateTransitionFunction, ObservationFunction, Policies...>>
{
public:
    typedef typename StateTransitionFunction::State State;
    typedef typename StateTransitionFunction::Input Input;
    typedef typename ObservationFunction::Obsrv Obsrv;
    typedef Gaussian<State> Belief;

    typedef GaussianFilter<
                StateTransitionFunction, ObservationFunction, Policies...
            > BaseGaussianFilter;

public:
    /**
     * \brief Creates a RobustGaussianFilter
     */
    template <typename ... SpecializationArgs>
    RobustGaussianFilter(const StateTransitionFunction& process_model,
                         const ObservationFunction& obsrv_model,
                         SpecializationArgs&& ... args)
        : gaussian_filter_(
              process_model,
              obsrv_model,
              std::forward<SpecializationArgs>(args)...)
    { }

    /**
     * \brief Overridable default destructor
     */
    virtual ~RobustGaussianFilter() { }

    /**
     * \copydoc FilterInterface::predict
     */
    virtual void predict(const Belief& prior_belief,
                         const Input& input,
                         Belief& predicted_belief)
    {
        gaussian_filter_.predict(prior_belief, input, predicted_belief);
    }

    /**
     * \copydoc FilterInterface::update
     */
    virtual void update(const Belief& predicted_belief,
                        const Obsrv& obsrv,
                        Belief& posterior_belief)
    {
        gaussian_filter_.update(predicted_belief, obsrv, posterior_belief);
    }

public: /* factory functions */
    virtual Belief create_belief() const
    {
        // note: do not simplify!
        auto belief = gaussian_filter_.create_belief();
        return belief;
    }

public: /* accessors & mutators */
    StateTransitionFunction& process_model()
    {
        return gaussian_filter_.process_model();
    }

    ObservationFunction& obsrv_model()
    {
        return gaussian_filter_.obsrv_model();
    }

    const StateTransitionFunction& process_model() const
    {
        return gaussian_filter_.process_model();
    }

    const ObservationFunction& obsrv_model() const
    {
        return gaussian_filter_.obsrv_model();
    }

    virtual std::string name() const
    {
        return "RobustGaussianFilter<"
                + this->list_arguments(
                            gaussian_filter_.name())
                + ">";
    }

    virtual std::string description() const
    {
        return "Robust GaussianFilter with"
                + this->list_descriptions(gaussian_filter_.description());
    }

protected:
    /** \cond internal */
    BaseGaussianFilter gaussian_filter_;
    /** \endcond */
};

}

#endif