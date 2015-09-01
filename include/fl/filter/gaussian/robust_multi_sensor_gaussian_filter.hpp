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
 * \file robust_multi_sensor_gaussian_filter.hpp
 * \date August 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

#include <fl/util/meta.hpp>
#include <fl/util/profiling.hpp>
#include <fl/util/traits.hpp>
#include <fl/filter/gaussian/sigma_point_additive_uncorrelated_update_policy.hpp>
#include <fl/filter/gaussian/sigma_point_additive_prediction_policy.hpp>
#include <fl/filter/gaussian/sigma_point_additive_update_policy.hpp>
#include <fl/filter/gaussian/sigma_point_prediction_policy.hpp>
#include <fl/filter/gaussian/sigma_point_update_policy.hpp>
#include <fl/model/observation/robust_feature_obsrv_model.hpp>

#include <fl/filter/gaussian/multi_sensor_gaussian_filter.hpp>

namespace fl
{


// Forward delcaration
template<
    typename StateTransitionFunction,
    typename JointObservationFunction,
    typename ... Policies
> class RobustMultiSensorGaussianFilter;

/**
 * \internal
 * \ingroup nonlinear_gaussian_filter
 *
 * Traits of the robust multi sensor GaussianFilter based on quadrature
 * with customizable policies, i.e implementations of the time and measurement
 * updates.
 */
template <
    typename StateTransitionFunction,
    typename JointObservationFunction,
    typename ... Policies
>
struct Traits<
          RobustMultiSensorGaussianFilter<
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
    typename ... Policies
>
class RobustMultiSensorGaussianFilter
    : public FilterInterface<
                 MultiSensorGaussianFilter<
                     StateTransitionFunction,
                     JointObservationFunction,
                     Policies...>>
{
public:
    typedef typename StateTransitionFunction::State State;
    typedef typename StateTransitionFunction::Input Input;
    typedef typename JointObservationFunction::Obsrv Obsrv;
    typedef Gaussian<State> Belief;

protected:
    /** \cond internal */

    typedef typename JointObservationFunction::LocalModel LocalModel;
    typedef typename JointObservationFunction::LocalObsrv LocalObsrv;
    typedef typename JointObservationFunction::LocalObsrvNoise LocalObsrvNoise;

    /**
     * \brief Local observation model used within the joint observation model
     */
    typedef typename JointObservationFunction::LocalModel LocalObsrvModel;

    /**
     * \brief Robust feature observation model type used internally.
     */
    typedef RobustFeatureObsrvModel<LocalObsrvModel> FeatureObsrvModel;

    /**
     * \brief Joint observation model
     */
    typedef JointObservationModel<
                MultipleOf<
                    FeatureObsrvModel,
                    JointObservationFunction::ModelCount>
            > JointFeatureObsrvModel;

    /**
     * \brief Internal generic multi-sensor GaussianFilter for nonlinear
     *        problems
     */
    typedef MultiSensorGaussianFilter<
                StateTransitionFunction, JointFeatureObsrvModel, Policies...
            > BaseGaussianFilter;

    /** \endcond */

public:
    /**
     * \brief Creates a RobustGaussianFilter
     */
    RobustMultiSensorGaussianFilter(
        const StateTransitionFunction& process_model,
        const JointObservationFunction& joint_obsrv_model)
        : gaussian_filter_(
              process_model,
              JointFeatureObsrvModel(
                  FeatureObsrvModel(joint_obsrv_model.local_obsrv_model()),
                  joint_obsrv_model.count_local_models()))
    { }

    /**
     * \brief Overridable default destructor
     */
    virtual ~RobustMultiSensorGaussianFilter() { }

    /**
     * \copydoc FilterInterface::predict
     */
    virtual void predict(const Belief& prior_belief,
                         const Input& input,
                         Belief& predicted_belief)
    {
        //gaussian_filter_.predict(prior_belief, input, predicted_belief);
    }

    /**
     * \copydoc FilterInterface::update
     */
    virtual void update(const Belief& predicted_belief,
                        const Obsrv& obsrv,
                        Belief& posterior_belief)
    {
//        typedef typename ObservationFunction::BodyObsrvModel::Noise Noise;

//        auto body_noise_distr = Gaussian<Noise>(obsrv_model()
//                                                    .body_model()
//                                                    .noise_dimension());

//        auto&& h = [&](const State& x, const Noise& w)
//        {
//           return obsrv_model().body_model().observation(x, w);
//        };

//        auto y_mean = typename FirstMomentOf<Obsrv>::Type();
//        auto y_cov = typename SecondMomentOf<Obsrv>::Type();

//        gaussian_filter_
//            .quadrature()
//            .integrate_moments(
//                h, predicted_belief, body_noise_distr, y_mean, y_cov);

//        gaussian_filter_
//            .obsrv_model()
//            .parameters(predicted_belief.mean(), y_mean, y_cov);

//        auto feature_y = gaussian_filter_
//                            .obsrv_model()
//                            .feature_obsrv(obsrv, predicted_belief.mean());

//        gaussian_filter_
//            .update(predicted_belief, feature_y, posterior_belief);
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

    JointObservationFunction& obsrv_model()
    {
        return gaussian_filter_.obsrv_model().embedded_obsrv_model();
    }

    JointFeatureObsrvModel& robust_joint_feature_obsrv_model()
    {
        return gaussian_filter_.obsrv_model();
    }

    const StateTransitionFunction& process_model() const
    {
        return gaussian_filter_.process_model();
    }

//    const JointObservationFunction& obsrv_model() const
//    {
//        return gaussian_filter_.obsrv_model().embedded_obsrv_model();
//    }

    const JointFeatureObsrvModel& robust_joint_feature_obsrv_model() const
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
