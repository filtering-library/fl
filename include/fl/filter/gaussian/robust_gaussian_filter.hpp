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
 * \file robust_gaussian_filter.hpp
 * \date August 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once


#include <fl/util/meta.hpp>
#include <fl/util/profiling.hpp>
#include <fl/util/traits.hpp>
#include <fl/filter/gaussian/gaussian_filter_nonlinear_generic.hpp>
#include <fl/filter/gaussian/update_policy/sigma_point_update_policy.hpp>
#include <fl/filter/gaussian/update_policy/sigma_point_additive_update_policy.hpp>
#include <fl/filter/gaussian/update_policy/sigma_point_additive_uncorrelated_update_policy.hpp>
#include <fl/filter/gaussian/prediction_policy/sigma_point_additive_prediction_policy.hpp>
#include <fl/filter/gaussian/prediction_policy/sigma_point_prediction_policy.hpp>
#include <fl/model/sensor/robust_sensor_function.hpp>

namespace fl
{


// Forward delcaration
template<
    typename TransitionFunction,
    typename SensorFunction,
    typename Quadrature,
    typename ... Policies
> class RobustGaussianFilter;

/**
 * \internal
 * \ingroup nonlinear_gaussian_filter
 *
 * Traits for generic GaussianFilter based on quadrature (numeric integration)
 * with customizable policies, i.e implementations of the time and measurement
 * updates.
 */
template <
    typename TransitionFunction,
    typename SensorFunction,
    typename Quadrature,
    typename ... Policies
>
struct Traits<
          RobustGaussianFilter<
              TransitionFunction,
              SensorFunction,
              Quadrature,
              Policies...>>
{
    typedef typename TransitionFunction::State State;
    typedef typename TransitionFunction::Input Input;
    typedef typename SensorFunction::Obsrv Obsrv;
    typedef Gaussian<State> Belief;
};

/**
 * \ingroup nonlinear_gaussian_filter
 * \brief Represents a Gaussian filter for nonlinear estimation using
 *        robustification in PAPER REF.
 *        Currently the filter assumes \a SensorFunction to be of the type
 *        a \a BodyTailSensor<BodtModelType, TailModelType>.
 *
 *        This filter may be used in the same fashion as any Gaussian filter.
 *        That is, the Quadrature and the prediction as well as the update
 *        policies are customizatble. The customization is identical to the
 *        Gaussian filter. For more details on custom policies
 *        see GaussianFilter!
 */
template<
    typename TransitionFunction,
    typename SensorFunction,
    typename Quadrature,
    typename ... Policies
>
class RobustGaussianFilter
    : public FilterInterface<
                 RobustGaussianFilter<
                     TransitionFunction,
                     SensorFunction,
                     Quadrature,
                     Policies...>>
{
public:
    typedef typename TransitionFunction::State State;
    typedef typename TransitionFunction::Input Input;
    typedef typename SensorFunction::Obsrv Obsrv;
    typedef Gaussian<State> Belief;

    /**
     * \brief Robust feature observation model type used internally.
     */
    typedef RobustSensor<SensorFunction> FeatureSensor;

    typedef GaussianFilter<
                TransitionFunction,
                FeatureSensor,
                Quadrature,
                Policies...
            > BaseGaussianFilter;

public:
    /**
     * \brief Creates a RobustGaussianFilter
     */
    template <typename ... SpecializationArgs>
    RobustGaussianFilter(const TransitionFunction& transition,
                         const SensorFunction& sensor,
                         const Quadrature& quadrature,
                         SpecializationArgs&& ... args)
        : sensor_(sensor),
          gaussian_filter_(
              transition,
              FeatureSensor(sensor_),
              quadrature,
              std::forward<SpecializationArgs>(args)...)
    { }

    /**
     * \brief Overridable default destructor
     */
    virtual ~RobustGaussianFilter() noexcept { }

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
        /* ------------------------------------------ */
        /* - Body model observation function lambda - */
        /* ------------------------------------------ */
        typedef typename SensorFunction::BodySensor::Noise Noise;

        auto&& h_body = [&](const State& x, const Noise& w)
        {
           return sensor().body_model().observation(x, w);
        };

        /* ------------------------------------------ */
        /* - Compute and set the feature parameter  - */
        /* - E[h_body(x, w)], Cov(h_body(x, w)) and - */
        /* - E[x]                                   - */
        /* ------------------------------------------ */
        auto y_mean = typename FirstMomentOf<Obsrv>::Type();
        auto y_cov = typename SecondMomentOf<Obsrv>::Type();
        quadrature()
            .integrate_moments(
                h_body,
                predicted_belief,
                Gaussian<Noise>(sensor().body_model().noise_dimension()),
                y_mean,
                y_cov);

        robust_sensor_function().mean_state(predicted_belief.mean());
        robust_sensor_function().body_moments(y_mean, y_cov);

        /* ------------------------------------------ */
        /* - compute feature using feature model    - */
        /* ------------------------------------------ */
        auto feature_y = robust_sensor_function().feature_obsrv(obsrv);

        /* ------------------------------------------ */
        /* - use the feature to update with a       - */
        /* - standard embedded GaussianFilter       - */
        /* ------------------------------------------ */
        gaussian_filter_
            .update(predicted_belief, feature_y, posterior_belief);
    }

public: /* factory functions */
    virtual Belief create_belief() const
    {
        auto belief = gaussian_filter_.create_belief();
        return belief; // RVO
    }

public: /* accessors & mutators */
    TransitionFunction& transition()
    {
        return gaussian_filter_.transition();
    }

    SensorFunction& sensor()
    {
        return gaussian_filter_.sensor().embedded_sensor();
    }

    Quadrature& quadrature()
    {
        return gaussian_filter_.quadrature();
    }

    const TransitionFunction& transition() const
    {
        return gaussian_filter_.transition();
    }

    const SensorFunction& sensor() const
    {
        return gaussian_filter_.sensor().embedded_sensor();
    }

    const Quadrature& quadrature() const
    {
        return gaussian_filter_.quadrature();
    }

    std::string name() const override
    {
        return "RobustGaussianFilter<"
                + this->list_arguments(
                            gaussian_filter_.name())
                + ">";
    }

    std::string description() const override
    {
        return "Robust GaussianFilter with"
                + this->list_descriptions(gaussian_filter_.description());
    }

protected:
    FeatureSensor& robust_sensor_function()
    {
        return gaussian_filter_.sensor();
    }

    typename SensorFunction::BodySensor body_model()
    {
        return sensor().body_model();
    }



protected:
    /** \cond internal */
    SensorFunction sensor_;
    BaseGaussianFilter gaussian_filter_;
    /** \endcond */
};

}


