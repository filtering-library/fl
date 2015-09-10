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
 * \file multi_sensor_sigma_point_update_policy.hpp
 * \date August 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

#include <errno.h>
#include <signal.h>
#include <assert.h>

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
template <typename...> class MultiSensorSigmaPointUpdatePolizzle;

/**
 * \internal
 */
template <
    typename SigmaPointQuadrature,
    typename NonJoinObservationModel
>
class MultiSensorSigmaPointUpdatePolizzle<
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


template <
    typename SigmaPointQuadrature,
    typename MultipleOfLocalObsrvModel
>
class MultiSensorSigmaPointUpdatePolizzle<
          SigmaPointQuadrature,
          JointObservationModel<MultipleOfLocalObsrvModel>>
    : public MultiSensorSigmaPointUpdatePolizzle<
                SigmaPointQuadrature,
                NonAdditive<JointObservationModel<MultipleOfLocalObsrvModel>>>
{ };

template <
    typename SigmaPointQuadrature,
    typename MultipleOfLocalObsrvModel
>
class MultiSensorSigmaPointUpdatePolizzle<
          SigmaPointQuadrature,
          NonAdditive<JointObservationModel<MultipleOfLocalObsrvModel>>>
    : public Descriptor
{
public:
    typedef JointObservationModel<MultipleOfLocalObsrvModel> JointModel;
    typedef typename MultipleOfLocalObsrvModel::Type LocalModel;

    typedef typename JointModel::State State;
    typedef typename JointModel::Obsrv Obsrv;
    typedef typename JointModel::Noise Noise;

    typedef typename Traits<JointModel>::LocalObsrv LocalObsrv;
    typedef typename Traits<JointModel>::LocalNoise LocalObsrvNoise;

    enum : signed int
    {
        NumberOfPoints = SigmaPointQuadrature::number_of_points(
                             JoinSizes<
                                 SizeOf<State>::Value,
                                 SizeOf<LocalObsrvNoise>::Value
                             >::Size)
    };

    typedef PointSet<State, NumberOfPoints> StatePointSet;
    typedef PointSet<LocalObsrv, NumberOfPoints> LocalObsrvPointSet;
    typedef PointSet<Vector1d, NumberOfPoints> LocalNoisePointSet;

    template <typename Belief>
    void operator()(JointModel& obsrv_function,
                    const SigmaPointQuadrature& quadrature,
                    const Belief& prior_belief,
                    const Obsrv& y,
                    Belief& posterior_belief)
    {
        auto& model = obsrv_function.local_obsrv_model();
        noise_distr_.dimension(model.noise_dimension());

        quadrature.transform_to_points(prior_belief, noise_distr_, p_X, p_Q);

//        auto&& h = [&](const State& x, const LocalObsrvNoise& w)
//        {
//            return model.observation(x, w);
//        };


        auto&& h_body = [&](const State& x, const fl::Vector1d& w)
        {
            return model.feature_obsrv(model.embedded_obsrv_model().body_model().observation(x, w));
        };


        auto&& h_tail = [&](const State& x, const fl::Vector1d& w)
        {
            return model.feature_obsrv(model.embedded_obsrv_model().tail_model().observation(x, w));
        };

        /// \todo write unit test why this *.center() followed by .points()
        /// introduces a bug
//        auto&& mu_x = p_X.center();
//        auto&& X = p_X.points();

        auto&& mu_x = p_X.mean();
        auto&& X = p_X.centered_points();

        auto&& W = p_X.covariance_weights_vector();
        auto c_xx = (X * W.asDiagonal() * X.transpose()).eval();
        auto c_xx_inv = c_xx.inverse().eval();

        auto C = c_xx_inv;
        auto D = State();
        D.setZero(mu_x.size());

        const int sensor_count = obsrv_function.count_local_models();
        const int dim_y = y.size() / sensor_count;// p_Y.dimension();

        assert(y.size() % sensor_count == 0);

        for (int i = 0; i < sensor_count; ++i)
        {
            bool valid = true;
            for (int k = i * dim_y; k < i * dim_y + dim_y; ++k)
            {
                if (!std::isfinite(y(k)))
                {
                    valid = false;
                    break;
                }
            }

            if (!valid) continue;

            model.id(i);



            quadrature.propergate_points(h_body, p_X, p_Q, p_Y);
            auto mu_y_body = p_Y.center();
            auto Y = p_Y.points();
            auto c_yy_body = (Y * W.asDiagonal() * Y.transpose()).eval();
            auto c_xy_body = (X * W.asDiagonal() * Y.transpose()).eval();



            quadrature.propergate_points(h_tail, p_X, p_Q, p_Y);
            auto mu_y_tail = p_Y.center();
            Y = p_Y.points();
            auto c_yy_tail = (Y * W.asDiagonal() * Y.transpose()).eval();
            auto c_xy_tail = (X * W.asDiagonal() * Y.transpose()).eval();




            auto weight = model.embedded_obsrv_model().weight_threshold();
            auto mu_y = (1.0 - weight) * mu_y_body + weight * mu_y_tail;
            auto c_yy = (1.0 - weight) * c_yy_body + weight * c_yy_tail;
            auto c_xy = (1.0 - weight) * c_xy_body + weight * c_xy_tail;


            auto c_yx = c_xy.transpose().eval();
            auto A_i = (c_yx * c_xx_inv).eval();
            auto c_yy_given_x = (c_yy - c_yx * c_xx_inv * c_xy).eval();

            auto innovation = (y.middleRows(i * dim_y, dim_y) - mu_y).eval();

            C += A_i.transpose() * solve(c_yy_given_x, A_i);
            D += A_i.transpose() * solve(c_yy_given_x, innovation);
        }

        posterior_belief.dimension(prior_belief.dimension());
        posterior_belief.covariance(C.inverse());
        posterior_belief.mean(mu_x + posterior_belief.covariance() * D);

//        if (posterior_belief.mean().topRows(3).array().abs().sum() > 10.)
//        {
//            PVT(y);
//            raise(SIGTRAP);
//        }
    }

    virtual std::string name() const
    {
        return "MultiSensorSigmaPointUpdatePolizzle<"
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
//    template <typename A, typename B>
//    auto cov(const A& a, const B& b)
//    -> typename std::remove_reference<decltype((a * b.transpose()).eval())>::type
//    {
//        auto&& W = p_X.covariance_weights_vector().asDiagonal();
//        auto c = (a * W * b.transpose()).eval();
//        return c;
//    }

protected:
    StatePointSet p_X;
    LocalNoisePointSet p_Q;
    LocalObsrvPointSet p_Y;

    Gaussian<Vector1d> noise_distr_;
};

}
