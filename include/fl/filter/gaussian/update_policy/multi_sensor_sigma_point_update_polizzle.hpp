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
 * \file multi_sensor_sigma_point_update_policy.hpp
 * \date August 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

#include <Eigen/Dense>

#include <fl/util/meta.hpp>
#include <fl/util/types.hpp>
#include <fl/util/traits.hpp>
#include <fl/util/descriptor.hpp>
#include <fl/model/sensor/joint_sensor_iid.hpp>
#include <fl/filter/gaussian/transform/point_set.hpp>
#include <fl/filter/gaussian/quadrature/sigma_point_quadrature.hpp>

namespace fl
{
// Forward declarations
template <typename...>
class MultiSensorSigmaPointUpdatePolizzle;

/**
 * \internal
 */
template <typename SigmaPointQuadrature, typename NonJoinSensor>
class MultiSensorSigmaPointUpdatePolizzle<SigmaPointQuadrature,
                                          NonJoinSensor>
{
    static_assert(
        std::is_base_of<internal::JointSensorIidType,
                        NonJoinSensor>::value,
        "\n\n\n"
        "====================================================================\n"
        "= Static Assert: You are using the wrong observation model type    =\n"
        "====================================================================\n"
        "  Observation model type must be a JointSensor<...>.      \n"
        "  For single observation model, use the regular Gaussian filter     \n"
        "  or the regular SigmaPointUpdatePolicy if you are specifying       \n"
        "  the update policy explicitly fo the GaussianFilter.               \n"
        "===================================================================="
        "\n");
};

template <typename SigmaPointQuadrature, typename MultipleOfLocalSensor>
class MultiSensorSigmaPointUpdatePolizzle<
    SigmaPointQuadrature,
    JointSensor<MultipleOfLocalSensor>>
    : public MultiSensorSigmaPointUpdatePolizzle<
          SigmaPointQuadrature,
          NonAdditive<JointSensor<MultipleOfLocalSensor>>>
{
};

template <typename SigmaPointQuadrature, typename MultipleOfLocalSensor>
class MultiSensorSigmaPointUpdatePolizzle<
    SigmaPointQuadrature,
    NonAdditive<JointSensor<MultipleOfLocalSensor>>>
    : public Descriptor
{
public:
    typedef JointSensor<MultipleOfLocalSensor> JointModel;
    typedef typename MultipleOfLocalSensor::Type FeatureSensor;
    typedef typename FeatureSensor::EmbeddedSensor BodyTailModel;
    typedef typename BodyTailModel::BodySensor BodyModel;
    typedef typename BodyTailModel::TailSensor TailModel;

    typedef typename JointModel::State State;
    typedef typename JointModel::Obsrv Obsrv;
    typedef typename JointModel::Noise Noise;

    typedef typename Traits<JointModel>::LocalObsrv LocalFeature;
    typedef Vector1d LocalObsrvNoise;

    template <typename Belief>
    void operator()(JointModel& obsrv_function,
                    const SigmaPointQuadrature& quadrature,
                    const Belief& prior_belief,
                    const Obsrv& y,
                    Belief& posterior_belief)
    {
        auto& feature_model = obsrv_function.local_sensor();
        auto& body_tail_model = feature_model.embedded_sensor();

        /* ------------------------------------------ */
        /* - Determine the number of quadrature     - */
        /* - points needed for the given quadrature - */
        /* - in conjunction with the joint Gaussian - */
        /* - p(State, LocalObsrvNoise)              - */
        /* ------------------------------------------ */
        enum : signed int
        {
            NumberOfPoints = SigmaPointQuadrature::number_of_points(
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
            prior_belief,
            Gaussian<LocalObsrvNoise>(feature_model.noise_dimension()),
            p_X,
            p_R);

        auto mu_x = p_X.mean();
        auto X = p_X.centered_points();

        auto W = p_X.covariance_weights_vector().asDiagonal();
        auto c_xx = (X * W * X.transpose()).eval();
        auto c_xx_inv = c_xx.inverse().eval();

        auto C = c_xx_inv;
        auto D = State();
        D.setZero(mu_x.size());

        const int sensor_count = obsrv_function.count_local_models();
        const int dim_y = y.size() / sensor_count;

        auto h_body = [&](const State& x, const typename BodyModel::Noise& w)
        {
            return feature_model.feature_obsrv(
                body_tail_model.body_model().observation(x, w));
        };
        auto h_tail = [&](const State& x, const typename TailModel::Noise& w)
        {
            return feature_model.feature_obsrv(
                body_tail_model.tail_model().observation(x, w));
        };
        PointSet<LocalFeature, NumberOfPoints> p_Y_body;
        PointSet<LocalFeature, NumberOfPoints> p_Y_tail;


        for (int i = 0; i < sensor_count; ++i)
        {
            // validate sensor value, i.e. make sure it is finite
            if (!is_valid(y, i * dim_y, i * dim_y + dim_y)) continue;

            feature_model.id(i);

            /* ------------------------------------------ */
            /* - Integrate body                         - */
            /* ------------------------------------------ */

            quadrature.propagate_points(h_body, p_X, p_R, p_Y_body);
            auto mu_y_body = p_Y_body.mean();

            // validate sensor value, i.e. make sure it is finite
            if (!is_valid(mu_y_body, 0, dim_y)) continue;

            auto Y_body = p_Y_body.centered_points();
            auto c_yy_body = (Y_body * W * Y_body.transpose()).eval();
            auto c_xy_body = (X * W * Y_body.transpose()).eval();

            /* ------------------------------------------ */
            /* - Integrate tail                         - */
            /* ------------------------------------------ */
            quadrature.propagate_points(h_tail, p_X, p_R, p_Y_tail);
            auto mu_y_tail = p_Y_tail.mean();
            auto Y_tail = p_Y_tail.centered_points();
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
            auto innovation = (y.middleRows(i * dim_y, dim_y) - mu_y).eval();

            C += A_i.transpose() * solve(c_yy_given_x, A_i);
            D += A_i.transpose() * solve(c_yy_given_x, innovation);
        }

        /* ------------------------------------------ */
        /* - Update belief according to PAPER REF   - */
        /* ------------------------------------------ */
        posterior_belief.dimension(prior_belief.dimension());
        posterior_belief.covariance(C.inverse());
        posterior_belief.mean(mu_x + posterior_belief.covariance() * D);
    }

    virtual std::string name() const
    {
        return "MultiSensorSigmaPointUpdatePolizzle<" +
               this->list_arguments("SigmaPointQuadrature",
                                    "NonAdditive<SensorFunction>") +
               ">";
    }

    virtual std::string description() const
    {
        return "Multi-Sensor Sigma Point based filter update policy "
               "for joint observation model of multiple local observation "
               "models with non-additive noise.";
    }

private:
    /**
     * \brief Checks whether all vector components within the range (start, end)
     *        are finiate, i.e. not NAN nor Inf.
     */
    template <typename Vector>
    bool is_valid(Vector&& vector, int start, int end) const
    {
        for (int k = start; k < end; ++k)
        {
            if (!std::isfinite(vector(k))) return false;
        }

        return true;
    }
};
}
