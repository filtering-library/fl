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
#include <fl/model/observation/robust_feature_obsrv_model.hpp>

namespace fl
{


// Forward delcaration
template<
    typename StateTransitionFunction,
    typename ObservationFunction,
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
    typename StateTransitionFunction,
    typename ObservationFunction,
    typename Quadrature,
    typename ... Policies
>
struct Traits<
          RobustGaussianFilter<
              StateTransitionFunction,
              ObservationFunction,
              Quadrature,
              Policies...>>
{
    typedef typename StateTransitionFunction::State State;
    typedef typename StateTransitionFunction::Input Input;
    typedef typename ObservationFunction::Obsrv Obsrv;
    typedef Gaussian<State> Belief;
};

/**
 * \ingroup nonlinear_gaussian_filter
 * \brief Represents a Gaussian filter for nonlinear estimation using
 *        robustification in PAPER REF.
 *        Currently the filter assumes \a ObservationFunction to be of the type
 *        a \a BodyTailObservationModel<BodtModelType, TailModelType>.
 *
 *        This filter may be used in the same fashion as any Gaussian filter.
 *        That is, the Quadrature and the prediction as well as the update
 *        policies are customizatble. The customization is identical to the
 *        Gaussian filter. For more details on custom policies
 *        see GaussianFilter!
 */
template<
    typename StateTransitionFunction,
    typename ObservationFunction,
    typename Quadrature,
    typename ... Policies
>
class RobustGaussianFilter
    : public FilterInterface<
                 RobustGaussianFilter<
                     StateTransitionFunction,
                     ObservationFunction,
                     Quadrature,
                     Policies...>>
{
public:
    typedef typename StateTransitionFunction::State State;
    typedef typename StateTransitionFunction::Input Input;
    typedef typename ObservationFunction::Obsrv Obsrv;
    typedef Gaussian<State> Belief;

    /**
     * \brief Robust feature observation model type used internally.
     */
    typedef RobustFeatureObsrvModel<ObservationFunction> FeatureObsrvModel;

    typedef GaussianFilter<
                StateTransitionFunction,
                FeatureObsrvModel,
                Quadrature,
                Policies...
            > BaseGaussianFilter;

public:
    /**
     * \brief Creates a RobustGaussianFilter
     */
    template <typename ... SpecializationArgs>
    RobustGaussianFilter(const StateTransitionFunction& process_model,
                         const ObservationFunction& obsrv_model,
                         const Quadrature& quadrature,
                         SpecializationArgs&& ... args)
        : obsrv_model_(obsrv_model),
          gaussian_filter_(
              process_model,
              FeatureObsrvModel(obsrv_model_),
              quadrature,
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
        /* ------------------------------------------ */
        /* - Body model observation function lambda - */
        /* ------------------------------------------ */
        typedef typename ObservationFunction::BodyObsrvModel::Noise Noise;

        auto&& h_body = [&](const State& x, const Noise& w)
        {
           return obsrv_model().body_model().observation(x, w);
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
                Gaussian<Noise>(obsrv_model().body_model().noise_dimension()),
                y_mean,
                y_cov);

        robust_feature_obsrv_model().mean_state(predicted_belief.mean());
        robust_feature_obsrv_model().body_moments(y_mean, y_cov);

        /* ------------------------------------------ */
        /* - compute feature using feature model    - */
        /* ------------------------------------------ */
        auto feature_y = robust_feature_obsrv_model().feature_obsrv(obsrv);

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
    StateTransitionFunction& process_model()
    {
        return gaussian_filter_.process_model();
    }

    ObservationFunction& obsrv_model()
    {
        return gaussian_filter_.obsrv_model().embedded_obsrv_model();
    }

    Quadrature& quadrature()
    {
        return gaussian_filter_.quadrature();
    }

    const StateTransitionFunction& process_model() const
    {
        return gaussian_filter_.process_model();
    }

    const ObservationFunction& obsrv_model() const
    {
        return gaussian_filter_.obsrv_model().embedded_obsrv_model();
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
    FeatureObsrvModel& robust_feature_obsrv_model()
    {
        return gaussian_filter_.obsrv_model();
    }

    typename ObservationFunction::BodyObsrvModel body_model()
    {
        return obsrv_model().body_model();
    }



protected:
    /** \cond internal */
    ObservationFunction obsrv_model_;
    BaseGaussianFilter gaussian_filter_;
    /** \endcond */
};

}


