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

namespace fl
{
// Forward delcaration
template <typename StateTransitionFunction,
          typename JointObsrvModel,
          typename Quadrature>
class RobustMultiSensorGaussianFilter;

/**
 * \internal
 * \ingroup nonlinear_gaussian_filter
 *
 * Traits of the robust multi sensor GaussianFilter based on quadrature
 * with customizable policies, i.e implementations of the time and measurement
 * updates.
 */
template <typename StateTransitionFunction,
          typename JointObsrvModel,
          typename Quadrature>
struct Traits<RobustMultiSensorGaussianFilter<StateTransitionFunction,
                                              JointObsrvModel,
                                              Quadrature>>
{
    typedef typename StateTransitionFunction::State State;
    typedef typename StateTransitionFunction::Input Input;
    typedef typename JointObsrvModel::Obsrv Obsrv;
    typedef Gaussian<State> Belief;
};

/**
 * \ingroup nonlinear_gaussian_filter
 */
template <typename StateTransitionFunction,
          typename JointObsrvModel,
          typename Quadrature>
class RobustMultiSensorGaussianFilter
    : public FilterInterface<
          RobustMultiSensorGaussianFilter<StateTransitionFunction,
                                          JointObsrvModel,
                                          Quadrature>>
{
public:
    typedef typename StateTransitionFunction::State State;
    typedef typename StateTransitionFunction::Input Input;
    typedef typename JointObsrvModel::Obsrv Obsrv;
    typedef Gaussian<State> Belief;

private:
    /** \cond internal */

    // Get the original local model types
    enum : signed int
    {
        ModelCount = JointObsrvModel::ModelCount
    };
    typedef typename JointObsrvModel::LocalModel BodyTailModel;

    // Define local feature observation model
    typedef RobustMultiSensorFeatureObsrvModel<BodyTailModel, ModelCount>
        FeatureObsrvModel;

    // Define robust joint feature observation model
    typedef JointObservationModel<MultipleOf<FeatureObsrvModel, ModelCount>>
        RobustJointFeatureObsrvModel;
    /**
     * \brief Internal generic multi-sensor GaussianFilter for nonlinear
     *        problems
     */
    typedef MultiSensorGaussianFilter<StateTransitionFunction,
                                      RobustJointFeatureObsrvModel,
                                      Quadrature>
        InternalMultiSensorGaussianFilter;

    /** \endcond */

public:
    /**
     * \brief Creates a RobustGaussianFilter
     */
    RobustMultiSensorGaussianFilter(
        const StateTransitionFunction& process_model,
        const JointObsrvModel& joint_obsrv_model,
        const Quadrature& quadrature)
        : joint_obsrv_model_(joint_obsrv_model),
          multi_sensor_gaussian_filter_(
              process_model,
              RobustJointFeatureObsrvModel(
                  FeatureObsrvModel(joint_obsrv_model_.local_obsrv_model(),
                                    joint_obsrv_model_.count_local_models()),
                  joint_obsrv_model_.count_local_models()),
              quadrature)
    {
    }

    /**
     * \brief Overridable default destructor
     */
    virtual ~RobustMultiSensorGaussianFilter() noexcept {}
    /**
     * \copydoc FilterInterface::predict
     */
    virtual void predict(const Belief& prior_belief,
                         const Input& input,
                         Belief& predicted_belief)
    {
        multi_sensor_gaussian_filter_.predict(
            prior_belief, input, predicted_belief);
    }

    /**
     * \copydoc FilterInterface::update
     */
    virtual void update(const Belief& predicted_belief,
                        const Obsrv& y,
                        Belief& posterior_belief)
    {
        auto& quadrature = multi_sensor_gaussian_filter_.quadrature();
        auto& feature_model = joint_feature_model().local_obsrv_model();
        auto& body_tail_model = feature_model.embedded_obsrv_model();

        typedef typename BodyTailModel::Obsrv LocalObsrv;
        typedef typename FeatureObsrvModel::Obsrv LocalFeature;
        typedef typename BodyTailModel::BodyObsrvModel::Noise LocalObsrvNoise;

        /* ------------------------------------------ */
        /* - Determine the number of quadrature     - */
        /* - points needed for the given quadrature - */
        /* - in conjunction with the joint Gaussian - */
        /* - p(State, LocalObsrvNoise)              - */
        /* ------------------------------------------ */
        enum : signed int
        {
            NumberOfPoints = Quadrature::number_of_points(
                JoinSizes<SizeOf<State>::Value,
                          SizeOf<LocalObsrvNoise>::Value>::Size)
        };

        /* ------------------------------------------ */
        /* - PointSets                              - */
        /* - [p_X, p_Q] ~ p(State, LocalObsrvNoise) - */
        /* ------------------------------------------ */
        PointSet<State, NumberOfPoints> p_X;
        PointSet<LocalObsrvNoise, NumberOfPoints> p_R;

        /* ------------------------------------------ */
        /* - Transform p(State, LocalObsrvNoise) to - */
        /* - point sets [p_X, p_Q]                  - */
        /* ------------------------------------------ */
        quadrature.transform_to_points(
            predicted_belief,
            Gaussian<LocalObsrvNoise>(body_tail_model.noise_dimension()),
            p_X,
            p_R);

        auto mu_x = p_X.mean();
        auto X = p_X.centered_points();

        auto W_vec = p_X.covariance_weights_vector();
        auto W = W_vec.asDiagonal();
        auto c_xx = (X * W * X.transpose()).eval();
        auto c_xx_inv = c_xx.inverse().eval();

        auto C = c_xx_inv;
        auto D = State();
        D.setZero(mu_x.size());

        const int sensor_count = joint_obsrv_model_.count_local_models();
        const int dim_y = body_tail_model.obsrv_dimension();

        auto h = [&](const State& x, const LocalObsrvNoise& w)
        {
            return body_tail_model.body_model().observation(x, w);
        };

        auto h_body = [&](const State& x, const LocalObsrvNoise& w)
        {
            return feature_model.feature_obsrv(
                body_tail_model.body_model().observation(x, w));
        };
        auto h_tail = [&](const State& x, const LocalObsrvNoise& w)
        {
            return feature_model.feature_obsrv(
                body_tail_model.tail_model().observation(x, w));
        };

        PointSet<LocalObsrv, NumberOfPoints> p_Y_body;
        PointSet<LocalFeature, NumberOfPoints> p_F_body;
        PointSet<LocalFeature, NumberOfPoints> p_F_tail;

        auto y_mean = typename FirstMomentOf<LocalObsrv>::Type();
        auto y_cov = typename SecondMomentOf<LocalObsrv>::Type();

        feature_model.mean_state(predicted_belief.mean());

        // compute body_tail_obsrv_model parameters
        for (int i = 0; i < sensor_count; ++i)
        {
            // validate sensor value, i.e. make sure it is finite
            if (!std::isfinite(y(i))) continue;

            feature_model.id(i);

            /* ------------------------------------------ */
            /* - Compute feature parameters             - */
            /* ------------------------------------------ */
            quadrature.propagate_points(h, p_X, p_R, p_Y_body);

            y_mean = p_Y_body.mean();
            if (!std::isfinite(y_mean(0))) continue;

            auto Z_c = p_Y_body.centered_points();
            y_cov = (Z_c * W * Z_c.transpose());

            // set the current sensor's parameter
            feature_model.body_moments(y_mean, y_cov);

            /* ------------------------------------------ */
            /* - Integrate body                         - */
            /* ------------------------------------------ */
            quadrature.propagate_points(h_body, p_X, p_R, p_F_body);
            auto mu_y_body = p_F_body.mean();

            // validate sensor value, i.e. make sure it is finite
            if (!std::isfinite(mu_y_body(0))) continue;

            auto Y_body = p_F_body.centered_points();
            auto c_yy_body = (Y_body * W * Y_body.transpose()).eval();
            auto c_xy_body = (X * W * Y_body.transpose()).eval();

            /* ------------------------------------------ */
            /* - Integrate tail                         - */
            /* ------------------------------------------ */
            quadrature.propagate_points(h_tail, p_X, p_R, p_F_tail);
            auto mu_y_tail = p_F_tail.mean();
            auto Y_tail = p_F_tail.centered_points();
            auto c_yy_tail = (Y_tail * W * Y_tail.transpose()).eval();
            auto c_xy_tail = (X * W * Y_tail.transpose()).eval();

            /* ------------------------------------------ */
            /* - Fuse and center                        - */
            /* ------------------------------------------ */
            auto w = body_tail_model.tail_weight();
            auto mu_y = ((1.0 - w) * mu_y_body + w * mu_y_tail).eval();

            // non centered moments
            auto m_yy_body =
                (c_yy_body + mu_y_body * mu_y_body.transpose()).eval();
            auto m_yy_tail =
                (c_yy_tail + mu_y_tail * mu_y_tail.transpose()).eval();
            auto m_yy = ((1.0 - w) * m_yy_body + w * m_yy_tail).eval();

            // center
            auto c_yy = (m_yy - mu_y * mu_y.transpose()).eval();
            auto c_xy = ((1.0 - w) * c_xy_body + w * c_xy_tail).eval();

            auto c_yx = c_xy.transpose().eval();
            auto A_i = (c_yx * c_xx_inv).eval();
            auto c_yy_given_x = (c_yy - c_yx * c_xx_inv * c_xy).eval();

            auto feature =
                feature_model.feature_obsrv(y.middleRows(i * dim_y, dim_y));
            auto innovation = (feature - mu_y).eval();

            C += A_i.transpose() * solve(c_yy_given_x, A_i);
            D += A_i.transpose() * solve(c_yy_given_x, innovation);
        }

        /* ------------------------------------------ */
        /* - Update belief according to PAPER REF   - */
        /* ------------------------------------------ */
        posterior_belief.dimension(predicted_belief.dimension());
        posterior_belief.covariance(C.inverse());
        posterior_belief.mean(mu_x + posterior_belief.covariance() * D);
    }

public: /* factory functions */
    virtual Belief create_belief() const
    {
        auto belief = multi_sensor_gaussian_filter_.create_belief();
        return belief;  // RVO
    }

public: /* accessors & mutators */
    StateTransitionFunction& process_model()
    {
        return multi_sensor_gaussian_filter_.process_model();
    }

    JointObsrvModel& obsrv_model() { return joint_obsrv_model_; }
    const StateTransitionFunction& process_model() const
    {
        return multi_sensor_gaussian_filter_.process_model();
    }

    const JointObsrvModel& obsrv_model() const { return joint_obsrv_model_; }
    std::string name() const override
    {
        return "RobustMultiSensorGaussianFilter<" +
               this->list_arguments(multi_sensor_gaussian_filter_.name()) + ">";
    }

    std::string description() const override
    {
        return "Robust multi-sensor GaussianFilter with" +
               this->list_descriptions(
                   multi_sensor_gaussian_filter_.description());
    }

protected:
    RobustJointFeatureObsrvModel& joint_feature_model()
    {
        return multi_sensor_gaussian_filter_.obsrv_model();
    }

protected:
    /** \cond internal */
    JointObsrvModel joint_obsrv_model_;
    InternalMultiSensorGaussianFilter multi_sensor_gaussian_filter_;
    /** \endcond */
};
}
