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
 * \file gaussian_filter_factroized_test.cpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

//#define EIGEN_NO_MALLOC

#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <fl/util/meta.hpp>
#include <fl/model/process/linear_process_model.hpp>
#include <fl/model/process/joint_process_model.hpp>
#include <fl/model/observation/linear_observation_model.hpp>

#include <fl/filter/filter_interface.hpp>
#include <fl/filter/gaussian/unscented_transform.hpp>
#include <fl/filter/gaussian/monte_carlo_transform.hpp>
#include <fl/filter/gaussian/feature_policy.hpp>
#include <fl/filter/gaussian/gaussian_filter_kf.hpp>
#include <fl/filter/gaussian/gaussian_filter_ukf.hpp>
#include <fl/filter/gaussian/gaussian_filter_factorized.hpp>


TEST(GaussianFilterFactorizedTests, vsKF)
{
    using namespace fl;
    /* ====================================================================== */
    /* = Setup                                                              = */
    /* ====================================================================== */

    /* -- Common setup ------------------------------------------------------ */

    double A_a = 1.0;
    double A_b = 1.0;
    double v_var_a = 0.0436345;
    double v_var_b = 0.05;
    double var_y = 0.0005634;

    typedef double Scalar;

    enum : signed int
    {
        Dim_a = 1,          // dimension of state a
        Dim_y_i = 1,        // dimension of a single measurement y_i
        Dim_b_i = 1,        // dimension of a parameter of an observation model
        SensorCount = 5,    // Number of sesors
        Dim_y = ExpandSizes<Dim_y_i, SensorCount>::Size,
        Dim_b = ExpandSizes<Dim_b_i, SensorCount>::Size,
        Dim_x = JoinSizes<Dim_a, Dim_b>::Size,

        // Compile-time values
        CTDim_a = Dim_a,
        CTDim_y_i = Dim_y_i,
        CTDim_b_i = Dim_b_i,
        CTSensorCount = SensorCount,
        CTDim_y = ExpandSizes<CTDim_y_i, CTSensorCount>::Size,
        CTDim_b = ExpandSizes<CTDim_b_i, CTSensorCount>::Size,
        CTDim_x = JoinSizes<CTDim_a, CTDim_b>::Size
    };

    /* -- Factorized definitions -------------------------------------------- */

    // local state a and observation y_i types
    typedef Eigen::Matrix<Scalar, CTDim_a, 1>   LocalState;
    typedef Eigen::Matrix<Scalar, CTDim_y_i, 1> LocalObsrv;

    // Local process and observation models the local observation model of  y_i
    typedef LinearGaussianProcessModel<LocalState> LocalProcessModel;
    typedef LinearGaussianObservationModel<LocalObsrv, LocalState> LocalObsrvModel;

    // define the parameter variate and its process model
    typedef typename Traits<LocalObsrvModel>::Param LocalParam;
    typedef LinearGaussianProcessModel<LocalParam> LocalParamModel;

    // Joint process model containing the joint of a and all multiples of b_i
    typedef JointProcessModel<
                LocalProcessModel,
                JointProcessModel<
                    MultipleOf<LocalParamModel, CTSensorCount>
                >
            > ProcessModel;

    // joint observation model
    typedef Adaptive<
                JointObservationModel<
                    MultipleOf<LocalObsrvModel, CTSensorCount>
                >
            > ObsrvModel;

    // filters sigma point transform
    //typedef UnscentedTransform Transform;
    typedef MonteCarloTransform<ConstantPointCountPolicy<1000>> Transform;
    //typedef IdentityFeaturePolicy<> FeaturePolicy;
    typedef SquaredFeaturePolicy<> FeaturePolicy;

    typedef GaussianFilter<
                LocalProcessModel,
                Join<
                    MultipleOf<
                       Adaptive<LocalObsrvModel, LocalParamModel>, CTSensorCount
                    >
                >,
                Transform,
                FeaturePolicy,
                Options<FactorizeParams>
            > FactorizedParamFilter;


    typedef Traits<FactorizedParamFilter>::State FPFState;
    typedef Traits<FactorizedParamFilter>::Obsrv FPFObsrv;
    typedef Traits<FactorizedParamFilter>::Input FPFInput;
    typedef Traits<FactorizedParamFilter>::LocalStateDistr FPFLocalStateDistr;
    typedef Traits<FactorizedParamFilter>::LocalParamDistr FPFLocalParamDistr;
    typedef Traits<FactorizedParamFilter>::JointParamDistr FPFJointParamDistr;
    typedef Traits<FactorizedParamFilter>::StateDistribution FPFStateDistribution;
    typedef Traits<FactorizedParamFilter>::FeatureMapping FeatureMapping;

    /* -- Joint Gaussian definitions ---------------------------------------- */

    typedef GaussianFilter<
                LocalProcessModel,
                Join<
                    MultipleOf<
                       Adaptive<LocalObsrvModel, LocalParamModel>, CTSensorCount
                    >
                >,
                Transform,
                FeaturePolicy,
                Options<NoOptions>
            > JointGaussianFilter;

    typedef Traits<JointGaussianFilter>::State JGFState;
    typedef Traits<JointGaussianFilter>::Obsrv JGFObsrv;
    typedef Traits<JointGaussianFilter>::Input JGFInput;
    typedef Traits<JointGaussianFilter>::StateDistribution JGFStateDistribution;
    typedef Traits<JointGaussianFilter>::FeatureMapping FeatureMapping;




    /* -- a Gaussian definitions ---------------------------------------------- */

//    typedef GaussianFilter<
//                LocalProcessModel,
//                JointObservationModel<MultipleOf<LocalObsrvModel, CTSensorCount>>,
//                Transform
//            > aGaussianFilter;

//    typedef Traits<aGaussianFilter>::State aState;
//    typedef Traits<aGaussianFilter>::Obsrv aObsrv;
//    typedef Traits<aGaussianFilter>::Input aInput;
//    typedef Traits<aGaussianFilter>::StateDistribution aStateDistribution;


    /* -- KalmanFilter definitions ------------------------------------------ */

    typedef typename Traits<ProcessModel>::State KFState;
    typedef Eigen::Matrix<Scalar, CTDim_y, 1> KFObsrv;

    // KF observation models
    typedef LinearGaussianProcessModel<KFState> LinearProcessModel;
    typedef LinearGaussianObservationModel<KFObsrv, KFState> LinearObsrvModel;

    // the kalman filter
    typedef GaussianFilter<LinearProcessModel, LinearObsrvModel> KalmanFilter;

    typedef Traits<KalmanFilter>::State KFState;
    typedef Traits<KalmanFilter>::Obsrv KFObsrv;
    typedef Traits<KalmanFilter>::Input KFInput;
    typedef Traits<KalmanFilter>::StateDistribution KFStateDistribution;

    /* ====================================================================== */
    /* = Create KalmanFilter                                                = */
    /* ====================================================================== */

    auto kalmanfilter = KalmanFilter(LinearProcessModel(Dim_x),
                                     LinearObsrvModel(Dim_y, Dim_x));

    auto& linear_process_model = kalmanfilter.process_model();
    auto& linear_obsrv_model  = kalmanfilter.obsrv_model();

    auto H_a = Traits<LocalObsrvModel>::SensorMatrix::Ones(Dim_y_i, Dim_a);
    auto H_b = Traits<LocalObsrvModel>::ParamMatrix::Ones(Dim_y_i, Dim_b_i);
    auto H = linear_obsrv_model.H();
    H.setZero();
    for (int i = 0; i < SensorCount; ++i)
    {
        H.block(i * H_a.rows(), 0, H_a.rows(), H_a.cols()) = H_a;

        H.block(i * H_a.rows(), H_a.cols() + i * H_b.cols(),
                H_b.rows(),     H_b.cols()) = H_b;
    }
    linear_obsrv_model.H(H);
    linear_obsrv_model.covariance(linear_obsrv_model.covariance() * var_y);

    {
        auto cov = linear_process_model.covariance();
        cov.block(0, 0, Dim_a, Dim_a) *= v_var_a;
        cov.block(Dim_a, Dim_a, Dim_b, Dim_b) *= v_var_b;
        linear_process_model.covariance(cov);

        auto A = linear_process_model.A();
        A.block(0, 0, Dim_a, Dim_a) *= A_a;
        A.block(Dim_a, Dim_a, Dim_b, Dim_b) *= A_b;
        linear_process_model.A(A);

        PV(linear_process_model.A());
        PV(linear_process_model.covariance());
    }

    auto kf_state = KFStateDistribution(Dim_x);

    /* ====================================================================== */
    /* = Create Factorized Parameter                                        = */
    /* ====================================================================== */

    auto fpfilter = FactorizedParamFilter(LocalProcessModel(Dim_a),
                                          LocalParamModel(Dim_b_i),
                                          LocalObsrvModel(Dim_y_i, Dim_a),
                                          Transform(),
                                          FeatureMapping(),
                                          SensorCount);
    {
        auto& local_process_model = fpfilter.local_process_model();
        auto& local_param_model = fpfilter.local_param_model();
        auto& local_obsrv_model = fpfilter.local_obsrv_model();

        local_process_model.covariance(local_process_model.covariance() * v_var_a);
        local_param_model.covariance(local_param_model.covariance() * v_var_b);
        local_obsrv_model.covariance(local_obsrv_model.covariance() * var_y);

        local_process_model.A(local_process_model.A() * A_a);
        local_param_model.A(local_param_model.A() * A_b);

        PV(local_process_model.A());
        PV(local_param_model.A());

        PV(local_process_model.covariance());
        PV(local_param_model.covariance());
    }

    auto fpf_state  = fpfilter.create_state_distribution();
//    auto fpf_state = FPFStateDistribution(
//                         FPFLocalStateDistr(Dim_a),
//                         FPFJointParamDistr(FPFLocalParamDistr(Dim_b_i), SensorCount));

    /* ====================================================================== */
    /* = Create Joint Gaussian                                              = */
    /* ====================================================================== */

    auto jgfilter = JointGaussianFilter(LocalProcessModel(Dim_a),
                                        LocalParamModel(Dim_b_i),
                                        LocalObsrvModel(Dim_y_i, Dim_a),
                                        Transform(),
                                        FeatureMapping(),
                                        SensorCount);
    {
        auto& local_process_model = std::get<0>(jgfilter.process_model().models());
        auto& local_param_model = std::get<1>(jgfilter.process_model().models()).local_process_model();
        auto& local_obsrv_model = jgfilter.obsrv_model().local_obsrv_model();

        local_process_model.covariance(local_process_model.covariance() * v_var_a);
        local_param_model.covariance(local_param_model.covariance() * v_var_b);
        local_obsrv_model.covariance(local_obsrv_model.covariance() * var_y);

        local_process_model.A(local_process_model.A() * A_a);
        local_param_model.A(local_param_model.A() * A_b);

        PV(local_process_model.A());
        PV(local_param_model.A());

        PV(local_process_model.covariance());
        PV(local_param_model.covariance());
    }

    auto jgf_state = jgfilter.create_state_distribution();


//    /* ====================================================================== */
//    /* = Create a Gaussian                                                  = */
//    /* ====================================================================== */

//    auto agfilter = aGaussianFilter(LocalProcessModel(Dim_a),
//                                   {LocalObsrvModel(Dim_y_i, Dim_a), SensorCount},
//                                   Transform());
//    {
//        auto& local_process_model = agfilter.process_model();
//        auto& local_obsrv_model = agfilter.obsrv_model().local_obsrv_model();

//        local_process_model.covariance(local_process_model.covariance() * v_var_a);
//        local_obsrv_model.covariance(local_obsrv_model.covariance() * var_y);

//        local_process_model.A(local_process_model.A() * A_a);

//        PV(local_process_model.A());
//        PV(local_process_model.covariance());
//    }

//    auto agf_state = agfilter.create_state_distribution();


    std::cout << "======================" << std::endl;

    EXPECT_TRUE((std::is_same<KFObsrv, FPFObsrv>::value));
    EXPECT_TRUE((std::is_same<KFObsrv, JGFObsrv>::value));

    PV(kf_state.covariance());
    PV(fpf_state.covariance());
    PV(jgf_state.covariance());

    PV(kf_state.mean());
    PV(fpf_state.mean());
    PV(jgf_state.mean());

    double max_steps = 100;
    double t = 0.0;

    double step = 2 * M_PI / max_steps;

    Gaussian<KFObsrv> rg(Dim_y);
    rg.covariance(rg.covariance() * var_y);

    for (int i = 0; i < int(max_steps); ++i)
    {

        t += step;

        KFObsrv y = FPFObsrv::Ones(Dim_y, 1);
        y = y * std::sin(t) + rg.sample();

        /* kalman filter */
        kalmanfilter.predict(1.0, KFInput(1, 1), kf_state, kf_state);
        kalmanfilter.update(y, kf_state, kf_state);

        auto kf_cov = kf_state.covariance();
        for (size_t i = 0; i < Dim_b; ++i)
        {
            kf_cov.block(0,                   Dim_a + i * Dim_b_i,
                         Dim_a + i * Dim_b_i, Dim_b_i).setZero();

            kf_cov.transpose()
                  .block(0,                   Dim_a + i * Dim_b_i,
                         Dim_a + i * Dim_b_i, Dim_b_i).setZero();
        }
        kf_state.covariance(kf_cov);

        /* factorized parameter filter */
        fpfilter.predict(1.0, FPFInput(Dim_b + 1, 1), fpf_state, fpf_state);
        fpfilter.update(y, fpf_state, fpf_state);

        /* joint gaussian filter */
        jgfilter.predict(1.0, FPFInput(Dim_b + 1, 1), jgf_state, jgf_state);
        jgfilter.update(y, jgf_state, jgf_state);

//        /* a gaussian filter */
//        agfilter.predict(1.0, aInput( 1, 1), agf_state, agf_state);
//        agfilter.update(y, agf_state, agf_state);


        auto cov = jgf_state.covariance();
        cov.setIdentity();
        cov.block(0, 0, Dim_a, Dim_a).setOnes();
        jgf_state.covariance(jgf_state.covariance().cwiseProduct(cov));

        ASSERT_TRUE(
            std::get<0>(fpf_state.distributions())
                .covariance()
                .ldlt()
                .isPositive());

        ASSERT_TRUE(kf_state.covariance().ldlt().isPositive());
        ASSERT_TRUE(jgf_state.covariance().ldlt().isPositive());
//        ASSERT_TRUE(agf_state.covariance().ldlt().isPositive());

//        EXPECT_TRUE(fpf_state.covariance().isApprox(kf_state.covariance(), 1.e-6));
//        EXPECT_TRUE(fpf_state.covariance().isApprox(jgf_state.covariance(), 1.e-6));
//        EXPECT_TRUE(fpf_state.mean().isApprox(kf_state.mean(), 1.e-6));
//        EXPECT_TRUE(fpf_state.mean().isApprox(jgf_state.mean(), 1.e-6));


//        PV(kf_state.covariance());
//        PV(fpf_state.covariance());
//        PV(jgf_state.covariance());
//        PV(agf_state.covariance());

//        PV(kf_state.mean());
//        PV(fpf_state.mean());
//        PV(jgf_state.mean());
//        PV(agf_state.mean());

        std::cout << "-------------" << std::endl;
    }

    PV(kf_state.covariance());
    PV(fpf_state.covariance());
    PV(jgf_state.covariance());
//    PV(agf_state.covariance());

    PV(kf_state.mean());
    PV(fpf_state.mean());
    PV(jgf_state.mean());
//    PV(agf_state.mean());

}





//TEST(GaussianFilterFactorizedTests, benchmark)
//{
//    using namespace fl;

//    constexpr int dim_a = 13;
//    constexpr int dim_y_i = 1;
//    constexpr int dim_b_i = 1;

//    constexpr int dpixels = 10000;
//    constexpr int pixels = -1;

//    double var_a = 0.4436345;
//    double var_y = 0.0035634;
//    double var_b = 0.2;

//    typedef Eigen::Matrix<double, dim_a, 1> State;
//    typedef Eigen::Matrix<double, dim_y_i, 1> Pixel;

//    typedef LinearGaussianProcessModel<State> ProcessModel;
//    typedef LinearGaussianObservationModel<Pixel, State> PixelModel;

//    typedef typename Traits<PixelModel>::Param Param;
//    typedef LinearGaussianProcessModel<Param> ParamModel;

//    // typedef MonteCarloTransform<> SigmaTransform;
//    typedef UnscentedTransform SigmaTransform;

//    typedef GaussianFilter<
//                JointProcessModel<
//                    ProcessModel,
//                    JointProcessModel<MultipleOf<ParamModel, pixels>>>,
//                Adaptive<JointObservationModel<MultipleOf<PixelModel, pixels>>>,
//                SigmaTransform,
//                Options<FactorizeParams>
//            > ExplicitFilter;

//    auto process_a  = ProcessModel(dim_a);
//    auto process_b  = ParamModel(dim_b_i);
//    auto obsrvmodel = PixelModel(dim_y_i, dim_a);

//    process_a.covariance(process_a.covariance() * var_a);

//    process_b.covariance(process_b.covariance() * var_b);
//    //process_b.A(process_b.A() * 0.);

//    obsrvmodel.covariance(obsrvmodel.covariance() * var_y);
//    auto H = obsrvmodel.H();
//    H.setRandom();
//    obsrvmodel.H(H);

//    ExplicitFilter x = ExplicitFilter(
//                           process_a,
//                           process_b,
//                           obsrvmodel,
//                           SigmaTransform(),
//                           dpixels);

//    typedef Traits<ExplicitFilter>::LocalStateDistr LocalStateDistr;
//    typedef Traits<ExplicitFilter>::LocalParamDistr LocalParamDistr;
//    typedef Traits<ExplicitFilter>::JointParamDistr JointParamDistr;

//    auto distr = Traits<ExplicitFilter>::StateDistribution(
//        LocalStateDistr(),
//        JointParamDistr(LocalParamDistr(), dpixels));

//    std::get<0>(distr.distributions()).mean(State::Ones(dim_a, 1) * 5);

//    for (int i = 0; i < 1; ++i)
//    {
//        x.predict(1.0, Traits<ExplicitFilter>::Input::Zero(dpixels + 1, 1), distr, distr);
//        x.update(Traits<ExplicitFilter>::Obsrv::Random(dpixels, 1), distr, distr);

////        if (!std::get<0>(distr.distributions()).covariance().ldlt().isPositive())
////        {
////            std::cout << "cov_aa not p.s.d at iterations " << i << std::endl;
////            break;
////        }

////        for (int k = 0; k < dpixels; ++k)
////        {
////            if (!std::get<1>(distr.distributions()).distribution(k).covariance().ldlt().isPositive())
////            {
////                std::cout << "cov_bb_" << k << " not p.s.d at iterations " << i << std::endl;
////                exit(1);
////            }
////        }
//    }


////    typedef GaussianFilter<
////                ProcessModel,
////                Join<MultipleOf<Adaptive<PixelModel, ParamModel>, pixels>>,
////                SigmaTransform,
////                Options<FactorizeParams>
////            > AutoFilter;

////    EXPECT_TRUE((std::is_base_of<ExplicitFilter, AutoFilter>::value));
////    ExplicitFilter y = AutoFilter(
////                           ProcessModel(13),
////                           ParamModel(1),
////                           PixelModel(1, 13),
////                           UnscentedTransform(),
////                           pixels);

//}
