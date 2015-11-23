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
#include <fl/model/observation/joint_observation_model_iid.hpp>
#include <fl/filter/gaussian/transform/point_set.hpp>
#include <fl/filter/gaussian/quadrature/sigma_point_quadrature.hpp>

namespace fl
{

// Forward declarations
template <typename...> class MultiSensorSigmaPointUpdatePolicy;

/**
 * \internal
 */
template <
    typename SigmaPointQuadrature,
    typename NonJoinObservationModel
>
class MultiSensorSigmaPointUpdatePolicy<
          SigmaPointQuadrature,
          NonJoinObservationModel>
{
    static_assert(
        std::is_base_of<
            internal::JointObservationModelIidType, NonJoinObservationModel
        >::value,
        "\n\n\n"
        "====================================================================\n"
        "= Static Assert: You are using the wrong observation model type    =\n"
        "====================================================================\n"
        "  Observation model type must be a JointObservationModel<...>.      \n"
        "  For single observation model, use the regular Gaussian filter     \n"
        "  or the regular SigmaPointUpdatePolicy if you are specifying       \n"
        "  the update policy explicitly fo the GaussianFilter.               \n"
        "====================================================================\n"
    );
};


/**
 * \brief Represents an update policy update for multiple sensors. This instance
 *        expects a \a JointObservationModel<>. The model is forwarded  as
 *        NonAdditive<JointObservationModel> to the actual
 *        implementation. In case you want to use the model as Additive, you may
 *        specify this explicitly using Additive<JointObservationModel> or
 *        UseAsAdditive<JointObservationModel>::Type
 */
template <
    typename SigmaPointQuadrature,
    typename MultipleOfLocalObsrvModel
>
class MultiSensorSigmaPointUpdatePolicy<
          SigmaPointQuadrature,
          JointObservationModel<MultipleOfLocalObsrvModel>>
    : public MultiSensorSigmaPointUpdatePolicy<
                SigmaPointQuadrature,
                NonAdditive<JointObservationModel<MultipleOfLocalObsrvModel>>>
{ };


/**
 * \brief Represents an update policy update functor for multiple sensors.
 *        This instance expects a \a NonAdditive<JointObservationModel<>>.
 *        The implementation exploits factorization in the joint observation.
 *        The update is performed for each sensor separately.
 */
template <
    typename SigmaPointQuadrature,
    typename MultipleOfLocalObsrvModel
>
class MultiSensorSigmaPointUpdatePolicy<
          SigmaPointQuadrature,
          NonAdditive<JointObservationModel<MultipleOfLocalObsrvModel>>>
    : public Descriptor
{
public:
    typedef JointObservationModel<MultipleOfLocalObsrvModel> JointModel;

    typedef typename JointModel::State State;
    typedef typename JointModel::Obsrv Obsrv;
    typedef typename JointModel::LocalObsrv LocalObsrv;
    typedef typename JointModel::LocalNoise LocalObsrvNoise;

    template <typename Belief>
    void operator()(JointModel& obsrv_function,
                    const SigmaPointQuadrature& quadrature,
                    const Belief& prior_belief,
                    const Obsrv& y,
                    Belief& posterior_belief)
    {
        auto& sensor_model = obsrv_function.local_obsrv_model();

        /* ------------------------------------------ */
        /* - Determine the number of quadrature     - */
        /* - points needed for the given quadrature - */
        /* - in conjunction with the joint Gaussian - */
        /* - p(State, LocalObsrvNoise)              - */
        /* ------------------------------------------ */
        enum : signed int
        {
            NumberOfPoints = SigmaPointQuadrature::number_of_points(
                                 JoinSizes<
                                     SizeOf<State>::Value,
                                     SizeOf<LocalObsrvNoise>::Value
                                 >::Size)
        };

        /* ------------------------------------------ */
        /* - PointSets                              - */
        /* - [p_X, p_Q] ~ p(State, LocalObsrvNoise) - */
        /* ------------------------------------------ */
        PointSet<State, NumberOfPoints> p_X;
        PointSet<LocalObsrvNoise, NumberOfPoints> p_Q;

        /* ------------------------------------------ */
        /* - PointSet [p_Y] = h(p_X, p_Q)           - */
        /* ------------------------------------------ */
        PointSet<LocalObsrv, NumberOfPoints> p_Y;

        /* ------------------------------------------ */
        /* - Transform p(State, LocalObsrvNoise) to - */
        /* - point sets [p_X, p_Q]                  - */
        /* ------------------------------------------ */
        quadrature.transform_to_points(
            prior_belief,
            Gaussian<LocalObsrvNoise>(sensor_model.noise_dimension()),
            p_X,
            p_Q);

        /* ------------------------------------------ */
        /* - Compute expected moments of the state  - */
        /* - E[X], Cov(X, X)                        - */
        /* ------------------------------------------ */
        auto W = p_X.covariance_weights_vector().asDiagonal();
        auto mu_x = p_X.mean();
        auto X = p_X.centered_points();
        auto c_xx_inv = (X * W * X.transpose()).inverse().eval();

        /* ------------------------------------------ */
        /* - Temporary accumulators which will be   - */
        /* - used to updated the belief             - */
        /* ------------------------------------------ */
        auto C = c_xx_inv;
        auto D = State();
        D.setZero(mu_x.size());

        const int sensor_count = obsrv_function.count_local_models();
        const int dim_y = y.size() / sensor_count;

        /* ------------------------------------------ */
        /* - lambda of the sensor observation       - */
        /* - function                               - */
        /* ------------------------------------------ */
        auto&& h = [&](const State& x, const LocalObsrvNoise& w)
        {
            return sensor_model.observation(x, w);
        };


        for (int i = 0; i < sensor_count; ++i)
        {
            // validate sensor value, i.e. make sure it is finite
            if (!is_valid(y, i * dim_y, i * dim_y + dim_y)) continue;

            // select current sensor and propagate the points through h(x, w)
            sensor_model.id(i);
            quadrature.propagate_points(h, p_X, p_Q, p_Y);

            // comute expected moments of the observation and validate
            auto mu_y = p_Y.mean();
            if (!is_valid(mu_y, 0, dim_y)) continue;

            // update accumulatorsa according to the equations in PAPER REF
            auto Y = p_Y.centered_points();
            auto c_xy = (X * W * Y.transpose()).eval();
            auto c_yx = c_xy.transpose().eval();
            auto A_i = (c_yx * c_xx_inv).eval();
            auto c_yy_given_x = (
                     (Y * W * Y.transpose()) - c_yx * c_xx_inv * c_xy
                 ).eval();

            auto innovation = (y.middleRows(i * dim_y, dim_y) - mu_y).eval();
            C += A_i.transpose() * solve(c_yy_given_x, A_i);
            D += A_i.transpose() * solve(c_yy_given_x, innovation);
        }

        /* ------------------------------------------ */
        /* - Update belief according to PAPER REF   - */
        /* ------------------------------------------ */
        // make sure the posterior has the correct dimension
        posterior_belief.dimension(prior_belief.dimension());
        posterior_belief.covariance(C.inverse());
        posterior_belief.mean(mu_x + posterior_belief.covariance() * D);
    }

    virtual std::string name() const
    {
        return "MultiSensorSigmaPointUpdatePolicy<"
                + this->list_arguments(
                       "SigmaPointQuadrature",
                       "NonAdditive<ObservationFunction>")
                + ">";
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
