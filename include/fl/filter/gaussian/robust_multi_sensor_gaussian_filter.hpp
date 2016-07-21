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
 * \file robust_multi_sensor_gaussian_filter.hpp
 * \date August 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once


#include <fl/util/meta.hpp>
#include <fl/util/profiling.hpp>
#include <fl/util/traits.hpp>

#include <fl/model/observation/robust_multi_sensor_feature_obsrv_model.hpp>

#include <fl/filter/gaussian/update_policy/sigma_point_update_policy.hpp>
#include <fl/filter/gaussian/update_policy/sigma_point_additive_update_policy.hpp>
#include <fl/filter/gaussian/update_policy/sigma_point_additive_uncorrelated_update_policy.hpp>
#include <fl/filter/gaussian/prediction_policy/sigma_point_prediction_policy.hpp>
#include <fl/filter/gaussian/prediction_policy/sigma_point_additive_prediction_policy.hpp>

#include <fl/filter/gaussian/multi_sensor_gaussian_filter.hpp>
#include <fl/filter/gaussian/X_robust_multi_sensor_gaussian_filter.hpp>

namespace fl
{


// Forward delcaration
template<
    typename TransitionFunction,
    typename ActualJointSensor,
    typename Quadrature
> class XRobustMultiSensorGaussianFilter;

/**
 * \internal
 * \ingroup nonlinear_gaussian_filter
 *
 * Traits of the robust multi sensor GaussianFilter based on quadrature
 * with customizable policies, i.e implementations of the time and measurement
 * updates.
 */
template <
    typename TransitionFunction,
    typename ActualJointSensor,
    typename Quadrature
>
struct Traits<
           XRobustMultiSensorGaussianFilter<
               TransitionFunction, ActualJointSensor, Quadrature>>
{
    typedef typename TransitionFunction::State State;
    typedef typename TransitionFunction::Input Input;
    typedef typename ActualJointSensor::Obsrv Obsrv;
    typedef Gaussian<State> Belief;
};


/**
 * \ingroup nonlinear_gaussian_filter
 */
template<
    typename TransitionFunction,
    typename ActualJointSensor,
    typename Quadrature
>
class XRobustMultiSensorGaussianFilter
    : public FilterInterface<
                XRobustMultiSensorGaussianFilter<
                    TransitionFunction, ActualJointSensor, Quadrature>>
{
public:
    typedef typename TransitionFunction::State State;
    typedef typename TransitionFunction::Input Input;
    typedef typename ActualJointSensor::Obsrv Obsrv;
    typedef Gaussian<State> Belief;

private:
    /** \cond internal */

    // Get the original local model types
    enum : signed int { ModelCount = ActualJointSensor::ModelCount };
    typedef typename ActualJointSensor::LocalModel PlainLocalModel;

    // Define local feature observation model
    typedef MultiRobustSensorFunction<
                PlainLocalModel, ModelCount
            > FeatureSensor;

    // Define robust joint feature observation model
    typedef JointSensor<
                MultipleOf<FeatureSensor, ModelCount>
            > RobustJointFeatureSensor;
    /**
     * \brief Internal generic multi-sensor GaussianFilter for nonlinear
     *        problems
     */
    typedef MultiSensorGaussianFilter<
                TransitionFunction,
                RobustJointFeatureSensor,
                Quadrature
            > InternalMultiSensorGaussianFilter;

    /** \endcond */

public:
    /**
     * \brief Creates a RobustGaussianFilter
     */
    XRobustMultiSensorGaussianFilter(
        const TransitionFunction& process_model,
        const ActualJointSensor& joint_obsrv_model,
        const Quadrature& quadrature)
        : joint_obsrv_model_(joint_obsrv_model),
          multi_sensor_gaussian_filter_(
              process_model,
              RobustJointFeatureSensor(
                  FeatureSensor(joint_obsrv_model_.local_obsrv_model(),
                                    joint_obsrv_model_.count_local_models()),
                  joint_obsrv_model_.count_local_models()),
              quadrature)
    { }

    /**
     * \brief Overridable default destructor
     */
    virtual ~XRobustMultiSensorGaussianFilter() noexcept { }

    /**
     * \copydoc FilterInterface::predict
     */
    virtual void predict(const Belief& prior_belief,
                         const Input& input,
                         Belief& predicted_belief)
    {
        multi_sensor_gaussian_filter_.predict(
            prior_belief,
            input,
            predicted_belief);
    }

    /**
     * \copydoc FilterInterface::update
     */
    virtual void update(const Belief& predicted_belief,
                        const Obsrv& y,
                        Belief& posterior_belief)
    {
        typedef typename PlainLocalModel::Obsrv PlainObsrv;
        typedef typename PlainLocalModel::BodySensor::Noise BodyNoise;
        typedef typename RobustJointFeatureSensor::Obsrv JointFeatureObsrv;

        auto joint_feature_y =
            JointFeatureObsrv(
                joint_feature_model()
                    .obsrv_dimension());
        joint_feature_y.setZero();

        auto local_body_noise_distr =
            Gaussian<BodyNoise>(
                obsrv_model()
                    .local_obsrv_model()
                    .body_model()
                    .noise_dimension());

        auto h = [&](const State& x, const BodyNoise& w)
        {
           return joint_obsrv_model_
                        .local_obsrv_model()
                        .body_model()
                        .observation(x, w);
        };

        enum : signed int
        {
            NumberOfPoints =
                Quadrature::number_of_points(
                    JoinSizes<
                        SizeOf<State>::Value,
                        SizeOf<BodyNoise>::Value
                    >::Value)
        };

        PointSet<State, NumberOfPoints> X;
        PointSet<BodyNoise, NumberOfPoints> R;
        PointSet<PlainObsrv, NumberOfPoints> Z;

        multi_sensor_gaussian_filter_
           .quadrature()
           .transform_to_points(predicted_belief, local_body_noise_distr, X, R);

        auto W = X.covariance_weights_vector().asDiagonal();

        auto y_mean = typename FirstMomentOf<PlainObsrv>::Type();
        auto y_cov = typename SecondMomentOf<PlainObsrv>::Type();

        auto& local_obsrv_model = obsrv_model().local_obsrv_model();
        auto& local_feature_model = joint_feature_model().local_obsrv_model();

        local_feature_model.mean_state(predicted_belief.mean());

        const int local_obsrv_dim = local_obsrv_model.obsrv_dimension();
        const int local_feature_dim = local_feature_model.obsrv_dimension();
        const int sensor_count = joint_obsrv_model_.count_local_models();

        low_level_obsrv_bg.setZero(sensor_count, 1);
        low_level_obsrv_fg.setZero(sensor_count, 1);
        low_level_obsrv_nan.setZero(sensor_count, 1);

        mean_obsrv.setZero(sensor_count);

        // compute body_tail_obsrv_model parameters
        for (int i = 0; i < sensor_count; ++i)
        {
            if (!std::isfinite(y(i)))
            {
                low_level_obsrv_nan(i) = 0.25;

                joint_feature_y(i * local_feature_dim) =
                    std::numeric_limits<Real>::quiet_NaN();
                continue;
            }

//            joint_obsrv_model_.local_obsrv_model().body_model().id(i);
            local_feature_model.id(i);
            multi_sensor_gaussian_filter_
                .quadrature()
                .propagate_points(h, X, R, Z);

            y_mean = Z.mean();
            mean_obsrv(i) = y_mean(0);
            //! \todo BG changes
            if (!std::isfinite(y_mean(0)))
            {
                low_level_obsrv_bg(i) = 0.50;

                joint_feature_y(i * local_feature_dim) =
                    std::numeric_limits<Real>::infinity();

                continue;
            }
            auto Z_c = Z.centered_points();
            y_cov = (Z_c * W * Z_c.transpose());

            // set the current sensor's parameter
            local_feature_model.body_moments(y_mean, y_cov, i);

            local_feature_model.id(i);
            auto feature = local_feature_model.feature_obsrv(
                        y.middleRows(i * local_obsrv_dim, local_obsrv_dim));

            joint_feature_y.middleRows(i * local_feature_dim, local_feature_dim) =
                    feature;

            low_level_obsrv_fg(i) = 0.75;

        }

        multi_sensor_gaussian_filter_
            .update(predicted_belief, joint_feature_y, posterior_belief);
    }

    Eigen::VectorXd mean_obsrv;

public: /* factory functions */
    virtual Belief create_belief() const
    {
        auto belief = multi_sensor_gaussian_filter_.create_belief();
        return belief; // RVO
    }


public: /* accessors & mutators */
    TransitionFunction& process_model()
    {
        return multi_sensor_gaussian_filter_.process_model();
    }

    ActualJointSensor& obsrv_model()
    {
        return joint_obsrv_model_;
    }

    const TransitionFunction& process_model() const
    {
        return multi_sensor_gaussian_filter_.process_model();
    }

    const ActualJointSensor& obsrv_model() const
    {
        return joint_obsrv_model_;
    }

    std::string name() const override
    {
        return "RobustMultiSensorGaussianFilter<"
                + this->list_arguments(multi_sensor_gaussian_filter_.name())
                + ">";
    }

    std::string description() const override
    {
        return "Robust multi-sensor GaussianFilter with"
                + this->list_descriptions(
                      multi_sensor_gaussian_filter_.description());
    }

protected:
    RobustJointFeatureSensor& joint_feature_model()
    {
        return multi_sensor_gaussian_filter_.obsrv_model();
    }

protected:
    /** \cond internal */
    ActualJointSensor joint_obsrv_model_;
    InternalMultiSensorGaussianFilter multi_sensor_gaussian_filter_;
    /** \endcond */

public:
    Eigen::VectorXd low_level_obsrv_bg;
    Eigen::VectorXd low_level_obsrv_fg;
    Eigen::VectorXd low_level_obsrv_nan;
};

}
