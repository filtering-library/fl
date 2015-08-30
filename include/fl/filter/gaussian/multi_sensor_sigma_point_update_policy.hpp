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

#ifndef FL__FILTER__GAUSSIAN__MULTI_SENSOR_SIGMA_POINT_UPDATE_POLICY_HPP
#define FL__FILTER__GAUSSIAN__MULTI_SENSOR_SIGMA_POINT_UPDATE_POLICY_HPP

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
    typedef typename MultipleOfLocalObsrvModel::Type LocalModel;

    typedef typename JointModel::State State;
    typedef typename JointModel::Obsrv Obsrv;
    typedef typename JointModel::Noise Noise;

    enum : signed int
    {
        NumberOfPoints = SigmaPointQuadrature::number_of_points(
                             JoinSizes<
                                 SizeOf<State>::Value,
                                 SizeOf<Noise>::Value
                             >::Size)
    };

    typedef PointSet<State, NumberOfPoints> StatePointSet;
    typedef PointSet<Noise, NumberOfPoints> NoisePointSet;
    typedef PointSet<Obsrv, NumberOfPoints> ObsrvPointSet;

    template <
        typename Belief
    >
    void operator()(const JointModel& obsrv_function,
                    const SigmaPointQuadrature& quadrature,
                    const Belief& prior_belief,
                    const Obsrv& obsrv,
                    Belief& posterior_belief)
    {
        // static_assert() is non-additive

        INIT_PROFILING
        noise_distr_.dimension(obsrv_function.noise_dimension());

        auto&& h = [&](const State& x, const Noise& w)
        {
           return obsrv_function.observation(x, w);
        };

        quadrature.propergate_gaussian(h, prior_belief, noise_distr_, X, Y, Z);
        MEASURE("Integrate");

        auto&& mu_y = Z.center();
        auto&& mu_x = X.center();
        auto&& Z_c = Z.points();
        auto&& X_c = X.points();
        auto&& W = X.covariance_weights_vector();
        auto cov_xx = (X_c * W.asDiagonal() * X_c.transpose()).eval();
        auto cov_xx_inv = (cov_xx.inverse()).eval();
        auto innovation = (obsrv - mu_y).eval();

        auto C = cov_xx_inv;
        auto D = mu_x;
        D.setZero();

        const int dim_Z_i = obsrv_function.local_obsrv_model().obsrv_dimension();
        const int obsrv_count = obsrv_function.count_local_models();

        MEASURE("Preparation");

//        for (int i = 0; i < obsrv_count; ++i)
//        {
//            auto Z_i = Z_c.middleRows(i * dim_Z_i, dim_Z_i).eval();
//            auto cov_xy_i = (X_c * W.asDiagonal() * Z_i.transpose()).eval();
//            auto cov_yy_i = (Z_i * W.asDiagonal() * Z_i.transpose()).eval();
//            auto innov_i  = innovation.middleRows(i * dim_Z_i, dim_Z_i).eval();

//            auto A_i = (cov_xy_i.transpose() * cov_xx_inv).eval();

//            auto cov_yy_i_inv_given_x =
//                (cov_yy_i - cov_xy_i.transpose() * cov_xx_inv * cov_xy_i)
//                    .inverse();

//            auto T = (A_i.transpose() * cov_yy_i_inv_given_x).eval();

//            C += T * A_i;
//            D += T * innov_i;
//        }


        auto Z_0 = Z_c.middleRows(0 * dim_Z_i, dim_Z_i).eval();
        auto cov_xy_0 = (X_c * W.asDiagonal() * Z_0.transpose()).eval();
        auto cov_yy_0 = (Z_0 * W.asDiagonal() * Z_0.transpose()).eval();
        auto innov_0  = innovation.middleRows(0 * dim_Z_i, dim_Z_i).eval();
        auto A_0 = (cov_xy_0.transpose() * cov_xx_inv).eval();
        auto cov_yy_i_inv_given_x =
            (cov_yy_0 - cov_xy_0.transpose() * cov_xx_inv * cov_xy_0)
                .inverse();
        auto T0 = (A_0.transpose() * cov_yy_i_inv_given_x).eval();
        MEASURE("Debugging temp preparation");


        for (int i = 0; i < obsrv_count; ++i) { auto Z_i = Z_c.middleRows(i * dim_Z_i, dim_Z_i).eval(); }
        MEASURE("auto Z_i = ...");

        for (int i = 0; i < obsrv_count; ++i) { auto cov_xy_i = (X_c * W.asDiagonal() * Z_0.transpose()).eval(); }
        MEASURE("auto cov_xy_i = ...");

        for (int i = 0; i < obsrv_count; ++i) { auto cov_yy_i = (Z_0 * W.asDiagonal() * Z_0.transpose()).eval(); }
        MEASURE("auto cov_yy_i = ...");

        for (int i = 0; i < obsrv_count; ++i){ auto innov_i  = innovation.middleRows(i * dim_Z_i, dim_Z_i).eval(); }
        MEASURE("auto innov_i = ...");

        for (int i = 0; i < obsrv_count; ++i){ auto A_i = (cov_xy_0.transpose() * cov_xx_inv).eval(); }
        MEASURE("auto A_i = ...");

        for (int i = 0; i < obsrv_count; ++i){ auto cov_yy_i_inv_given_x =
                    (cov_yy_0 - cov_xy_0.transpose() * cov_xx_inv * cov_xy_0)
                        .inverse(); }
        MEASURE("auto cov_yy_i_inv_given_x = ...");

        for (int i = 0; i < obsrv_count; ++i){ auto T = (A_0.transpose() * cov_yy_i_inv_given_x).eval(); }
        MEASURE("auto T = ...");

        for (int i = 0; i < obsrv_count; ++i){ C += T0 * A_0; }
        MEASURE("auto C = ...");

        for (int i = 0; i < obsrv_count; ++i){ D += T0 * innov_0; }
        MEASURE("auto D = ...");



        posterior_belief.covariance(C.inverse());
        MEASURE("posterior_belief.covariance(C.inverse())");
        posterior_belief.mean(mu_x + posterior_belief.covariance() * D);
        MEASURE("posterior_belief.mean(mu_x + posterior_belief.covariance() * D)");

        std::cout << "==================================================================" << std::endl;
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

protected:
    StatePointSet X;
    NoisePointSet Y;
    ObsrvPointSet Z;
    Gaussian<Noise> noise_distr_;
};

}

#endif
