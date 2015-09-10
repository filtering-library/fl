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
        LocalObsrvPointSet p_Y;
        Gaussian<LocalObsrvNoise> noise_distr_;


        auto& model = obsrv_function.local_obsrv_model();
        noise_distr_.dimension(model.noise_dimension());

        quadrature.transform_to_points(prior_belief, noise_distr_, p_X, p_Q);


        // create noise and state samples --------------------------------------
        PointSet<Vector1d, NumberOfPoints> p_Q_partial;
        auto&& W_body = p_X.covariance_weights_vector();
        auto&& W_tail = p_X.covariance_weights_vector();

        for(int i = 0; i < NumberOfPoints; i++)
        {
            p_Q_partial.point(i, p_Q.point(i).topRows(1), p_Q.weight(i));

            Real w_normal = p_Q.point(i).bottomRows(1)(0);
            Real w_uniform = fl::normal_to_uniform(w_normal);

            if(w_uniform > model.embedded_obsrv_model().weight_threshold())
            {
                W_body(i) = 0;
            }
            else
            {
                W_tail(i) = 0;
            }
        }
        // ---------------------------------------------------------------------




        auto&& h = [&](const State& x, const LocalObsrvNoise& w)
        {
            return model.observation(x, w);
        };

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

            quadrature.propagate_points(h, p_X, p_Q, p_Y);

            auto mu_y = p_Y.mean();

            valid = true;
            for (int k = 0; k < dim_y; ++k)
            {
                if (!std::isfinite(mu_y(k)))
                {
                    valid = false;
                    break;
                }
            }
            if (!valid) continue;

            auto Y = p_Y.centered_points();
            auto c_yy = (Y * W.asDiagonal() * Y.transpose()).eval();
            auto c_xy = (X * W.asDiagonal() * Y.transpose()).eval();



            // integrate body --------------------------------------------------
            auto&& h_body = [&](const State& x, const fl::Vector1d& w)
            {
                auto obsrv =
                    model.embedded_obsrv_model().body_model().observation(x, w);
                auto feature = model.feature_obsrv(obsrv);
                return feature;
            };
            PointSet<LocalObsrv, NumberOfPoints> p_Y_body;
            quadrature.propagate_points(h_body, p_X, p_Q_partial, p_Y_body);

            Eigen::Vector3d mu_y_body =
                    (p_Y_body.points() * W_body.asDiagonal()).rowwise().sum();

//            Eigen::Vector3d mu_y_body = p_Y_body.mean();
            Eigen::Matrix<double, 3, NumberOfPoints> Y_body =
                    p_Y_body.centered_points();
            Eigen::Matrix<double, 3, 3> c_yy_body =
                    (Y_body * W_body.asDiagonal() * Y_body.transpose()).eval();
            Eigen::Matrix<double, 6, 3> c_xy_body =
                    (X * W_body.asDiagonal() * Y_body.transpose()).eval();
            // -----------------------------------------------------------------

            // integrate tail --------------------------------------------------
            auto&& h_tail = [&](const State& x, const fl::Vector1d& w)
            {
                return model.feature_obsrv(
                   model.embedded_obsrv_model().tail_model().observation(x, w));
            };
            PointSet<LocalObsrv, NumberOfPoints> p_Y_tail;
            quadrature.propagate_points(h_tail, p_X, p_Q_partial, p_Y_tail);


            Eigen::Vector3d mu_y_tail =
                    (p_Y_tail.points() * W_tail.asDiagonal()).rowwise().sum();

//            Eigen::Vector3d mu_y_tail = p_Y_tail.mean();
            Eigen::Matrix<double, 3, NumberOfPoints> Y_tail =
                                                p_Y_tail.centered_points();
            Eigen::Matrix<double, 3, 3> c_yy_tail =
                          (Y_tail * W_tail.asDiagonal() * Y_tail.transpose()).eval();
            Eigen::Matrix<double, 6, 3> c_xy_tail =
                            (X * W_tail.asDiagonal() * Y_tail.transpose()).eval();
            // -----------------------------------------------------------------


//            // sum it up -------------------------------------------------------
//            double weight = model.embedded_obsrv_model().weight_threshold();
//            mu_y = ((1.0 - weight) * mu_y_body + weight * mu_y_tail).eval();
//            c_yy = ((1.0 - weight) * c_yy_body + weight * c_yy_tail).eval();
//            c_xy = ((1.0 - weight) * c_xy_body + weight * c_xy_tail).eval();
//            // -----------------------------------------------------------------


//            PF(mu_y.transpose());
//            PF(mu_y_body.transpose());

//            PF(c_yy);
//            PF(c_yy_body);

//            PF(c_xy);
//            PF(c_xy_body);


//            if(!mu_y.isApprox(mu_y_body))
//            {
//                PF(mu_y.transpose());
//                PF(mu_y_body.transpose());

//                std::cout << "py: " << std::endl << Y << std::endl;
//                std::cout << "py_body: " << std::endl << Y_body << std::endl;


//                std::cout << "py: " << std::endl << Y.rowwise().sum() << std::endl;
//                std::cout << "py_body: " << std::endl << Y_body.rowwise().sum() << std::endl;


//                PF(p_Y_body.points());
//                exit(-1);
//            }

//            if(!c_yy.isApprox(c_yy_body))
//            {
//                PF(c_yy);
//                PF(c_yy_body);
//                exit(-1);

//            }
//            if(!c_xy.isApprox(c_xy_body))
//            {
//                PF(c_xy);
//                PF(c_xy_body);
//                exit(-1);

//            }




            if(!mu_y.isApprox(mu_y_body + mu_y_tail))
            {
                std::cout << "mean " << std::endl
                          << mu_y.transpose() << std::endl;
                std::cout << "composed mean " << std::endl
                          << (mu_y_body + mu_y_tail).transpose() << std::endl;



                std::cout << "weighted tail points" << std::endl
                     << (p_Y_tail.points() * W_tail.asDiagonal()) << std::endl;

                std::cout << "weighted body points" << std::endl
                     << (p_Y_body.points() * W_body.asDiagonal()) << std::endl;


                std::cout << "weighted points" << std::endl
                     << (p_Y.points() * W.asDiagonal()) << std::endl;

                exit(-1);
            }

            if(!c_yy.isApprox(c_yy_body + c_yy_tail))
            {
                PF(c_yy);
                PF(c_yy_tail);
                exit(-1);

            }
            if(!c_xy.isApprox(c_xy_body + c_xy_tail))
            {
                PF(c_xy);
                PF(c_xy_tail);
                exit(-1);

            }





//            // sum it up -------------------------------------------------------
//            mu_y =  mu_y_body;
//            c_yy =  c_yy_body;
//            c_xy = c_xy_body;
//            // -----------------------------------------------------------------















            auto c_yx = c_xy.transpose().eval();
            auto A_i = (c_yx * c_xx_inv).eval();
            auto c_yy_given_x = (c_yy - c_yx * c_xx_inv * c_xy).eval();

            auto innovation = (y.middleRows(i * dim_y, dim_y) - mu_y).eval();

            Eigen::MatrixXd c_yy_given_x_inv_A_i =
                c_yy_given_x.colPivHouseholderQr().solve(A_i).eval();
            Eigen::MatrixXd c_yy_given_x_inv_innovation =
                c_yy_given_x.colPivHouseholderQr().solve(innovation).eval();

            C += A_i.transpose() * c_yy_given_x_inv_A_i;
            D += A_i.transpose() * c_yy_given_x_inv_innovation;

            //            C += A_i.transpose() * solve(c_yy_given_x, A_i);
            //            D += A_i.transpose() * solve(c_yy_given_x, innovation);

//            break_on_fail(
//                D.array().square().sum() < 1.e9);
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

