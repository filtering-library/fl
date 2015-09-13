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
//    typedef typename Traits<JointModel>::LocalNoise LocalObsrvNoise;
    typedef Vector1d LocalObsrvNoise;


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
    typedef PointSet<LocalObsrvNoise, NumberOfPoints> LocalNoisePointSet;

    template <typename Belief>
    void operator()(JointModel& obsrv_function,
                    const SigmaPointQuadrature& quadrature,
                    const Belief& prior_belief,
                    const Obsrv& y,
                    Belief& posterior_belief)
    {
        StatePointSet p_X;
        LocalNoisePointSet p_Q;
        Gaussian<LocalObsrvNoise> noise_distr;

        /// todo: we might have to set the size of the noise distr;

        auto& model = obsrv_function.local_obsrv_model();
        quadrature.transform_to_points(prior_belief, noise_distr, p_X, p_Q);

        auto mu_x = p_X.mean();
        auto X = p_X.centered_points();

        auto W = p_X.covariance_weights_vector();
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

            // integrate body --------------------------------------------------
            auto h_body = [&](const State& x,
                              const typename LocalModel::EmbeddedObsrvModel::BodyObsrvModel::Noise& w)
            {
                auto obsrv =
                    model.embedded_obsrv_model().body_model().observation(x, w);
                auto feature = model.feature_obsrv(obsrv);
                return feature;
            };
            PointSet<LocalObsrv, NumberOfPoints> p_Y_body;
            quadrature.propagate_points(h_body, p_X, p_Q, p_Y_body);



            auto mu_y_body = p_Y_body.mean();
            valid = true;
            for (int k = 0; k < dim_y; ++k)
            {
                if (!std::isfinite(mu_y_body(k)))
                {
                    valid = false;
                    break;
                }
            }
            if (!valid) continue;




            auto Y_body = p_Y_body.centered_points();

            auto c_yy_body =
                    (Y_body * W.asDiagonal() * Y_body.transpose()).eval();
            auto c_xy_body =
                    (X * W.asDiagonal() * Y_body.transpose()).eval();



            // integrate tail --------------------------------------------------
            auto h_tail = [&](const State& x,
                    const typename LocalModel::EmbeddedObsrvModel::TailObsrvModel::Noise& w)
            {
                auto obsrv = model.embedded_obsrv_model().tail_model().observation(x, w);
                auto feature = model.feature_obsrv(obsrv);
                return feature;
            };
            PointSet<LocalObsrv, NumberOfPoints> p_Y_tail;
            quadrature.propagate_points(h_tail, p_X, p_Q, p_Y_tail);


            auto mu_y_tail = p_Y_tail.mean();
            auto Y_tail = p_Y_tail.centered_points();


            auto c_yy_tail =
                          (Y_tail * W.asDiagonal() * Y_tail.transpose()).eval();
            auto c_xy_tail =
                            (X * W.asDiagonal() * Y_tail.transpose()).eval();
            // -----------------------------------------------------------------


            // fuse ------------------------------------------------------------
            Real t = model.embedded_obsrv_model().weight_threshold();
            Real b = 1.0 - t;
            auto mu_y = (b * mu_y_body + t * mu_y_tail).eval();


            // non centered moments
            auto m_yy_body = c_yy_body + mu_y_body * mu_y_body.transpose();
            auto m_yy_tail = c_yy_tail + mu_y_tail * mu_y_tail.transpose();
            auto m_yy = b * m_yy_body + t * m_yy_tail;

            // center
            auto c_yy = (m_yy - mu_y * mu_y.transpose());



            auto c_xy = (b * c_xy_body + t * c_xy_tail).eval();

            /// todo: this extra shit should not be necessesary and is only for
            /// comparsiosn
//            c_xy = c_xy - b * (mu_x_body - mu_x) * (mu_y - mu_y_body).transpose();
//            c_xy = c_xy - t * (mu_x_tail - mu_x) * (mu_y - mu_y_tail).transpose();



            auto c_yx = c_xy.transpose().eval();
            auto A_i = (c_yx * c_xx_inv).eval();
            auto c_yy_given_x = (c_yy - c_yx * c_xx_inv * c_xy).eval();

            auto innovation = (y.middleRows(i * dim_y, dim_y) - mu_y).eval();

            auto c_yy_given_x_inv_A_i =
                c_yy_given_x.colPivHouseholderQr().solve(A_i).eval();
            auto c_yy_given_x_inv_innovation =
                c_yy_given_x.colPivHouseholderQr().solve(innovation).eval();

            C += A_i.transpose() * c_yy_given_x_inv_A_i;
            auto delta = (A_i.transpose() * c_yy_given_x_inv_innovation).eval();

            D += delta;
        }


        posterior_belief.dimension(prior_belief.dimension());
        posterior_belief.covariance(C.inverse());
        posterior_belief.mean(mu_x + posterior_belief.covariance() * D);
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
};

}

