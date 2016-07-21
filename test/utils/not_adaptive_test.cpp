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
 * \file not_adaptivetest.hpp
 * \date March 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <type_traits>

#include <fl/util/traits.hpp>

#include <fl/util/meta.hpp>
#include <fl/model/sensor/linear_sensor.hpp>
#include <fl/model/sensor/joint_sensor.hpp>
#include <fl/model/process/linear_transition.hpp>

TEST(NotAdaptiveTests, fixed_traits)
{
    using namespace fl;

    typedef LinearGaussianObservationModel<
                Eigen::Matrix<double, 2, 1>,
                Eigen::Matrix<double, 3, 1>
            > ObsrvModel;

    typedef NotAdaptive<ObsrvModel> NAObsrvModel;

    EXPECT_TRUE((
        std::is_same<
            Traits<ObsrvModel>::Obsrv,
            Traits<NAObsrvModel>::Obsrv>::value));

    EXPECT_TRUE((
        std::is_same<
            Traits<ObsrvModel>::Noise,
            Traits<NAObsrvModel>::Noise>::value));

    EXPECT_TRUE((
        std::is_same<
            Traits<ObsrvModel>::State,
            Traits<NAObsrvModel>::State>::value));

    EXPECT_EQ(Traits<NAObsrvModel>::Param::SizeAtCompileTime, 0);
}

TEST(NotAdaptiveTests, dynamic_traits)
{
    using namespace fl;

    typedef LinearGaussianObservationModel<
                Eigen::Matrix<double, -1, 1>,
                Eigen::Matrix<double, -1, 1>
            > ObsrvModel;

    typedef NotAdaptive<ObsrvModel> NAObsrvModel;

    EXPECT_TRUE((
        std::is_same<
            Traits<ObsrvModel>::Obsrv,
            Traits<NAObsrvModel>::Obsrv>::value));

    EXPECT_TRUE((
        std::is_same<
            Traits<ObsrvModel>::Noise,
            Traits<NAObsrvModel>::Noise>::value));

    EXPECT_TRUE((
        std::is_same<
            Traits<ObsrvModel>::State,
            Traits<NAObsrvModel>::State>::value));

    EXPECT_EQ(Traits<NAObsrvModel>::Param::SizeAtCompileTime, 0);
}

TEST(NotAdaptiveTests, fixed_traits_of_adaptive_model)
{
    using namespace fl;

    typedef LinearGaussianObservationModel<
                Eigen::Matrix<double, 2, 1>,
                Eigen::Matrix<double, 3, 1>
            > LocalObsrvModel;

    typedef JointObservationModel<MultipleOf<LocalObsrvModel, 3>> ObsrvModel;
    typedef NotAdaptive<ObsrvModel> NAObsrvModel;

    EXPECT_TRUE((
        std::is_same<
            Traits<ObsrvModel>::Obsrv,
            Traits<NAObsrvModel>::Obsrv>::value));

    EXPECT_TRUE((
        std::is_same<
            Traits<ObsrvModel>::Noise,
            Traits<NAObsrvModel>::Noise>::value));

    EXPECT_TRUE((
        std::is_same<
            Traits<ObsrvModel>::State,
            Traits<NAObsrvModel>::State>::value));

    EXPECT_EQ(Traits<NAObsrvModel>::Param::SizeAtCompileTime, 0);
}

TEST(NotAdaptiveTests, Fixed_LinearObsrvModel)
{
    using namespace fl;

    typedef LinearGaussianObservationModel<
                Eigen::Matrix<double, 2, 1>,
                Eigen::Matrix<double, 3, 1>
            > ObsrvModel;

    typedef NotAdaptive<ObsrvModel> NAObsrvModel;

    typedef typename Traits<ObsrvModel>::Obsrv Obsrv;
    typedef typename Traits<ObsrvModel>::State State;
    typedef typename Traits<ObsrvModel>::Noise Noise;
    typedef typename Traits<ObsrvModel>::Noise Noise;

    ObsrvModel sensor;
    NAObsrvModel na_sensor = NAObsrvModel(ObsrvModel());

    State state = State::Random();
    Noise noise = Noise::Random();


    EXPECT_TRUE(
        fl::are_similar(
            sensor.predict_obsrv(state, noise, 1.0),
            na_sensor.predict_obsrv(state, noise, 1.0)));

    EXPECT_EQ(sensor.predict_obsrv(state, noise, 1.0).rows(),
              na_sensor.predict_obsrv(state, noise, 1.0).rows());
}

TEST(NotAdaptiveTests, Dynamic_LinearObsrvModel)
{
    using namespace fl;

    typedef Eigen::Matrix<double, -1, 1> Obsrv;
    typedef Eigen::Matrix<double, -1, 1> State;

    typedef LinearGaussianObservationModel<Obsrv, State> ObsrvModel;
    typedef NotAdaptive<ObsrvModel> NAObsrvModel;

    typedef typename Traits<ObsrvModel>::Noise Noise;

    ObsrvModel sensor = ObsrvModel(2, 3);
    NAObsrvModel na_sensor = NAObsrvModel(ObsrvModel(2, 3));

    State state = State::Random(3);
    Noise noise = Noise::Random(2);

    EXPECT_TRUE(
        fl::are_similar(
            sensor.predict_obsrv(state, noise, 1.0),
            na_sensor.predict_obsrv(state, noise, 1.0)));

    EXPECT_EQ(sensor.predict_obsrv(state, noise, 1.0).rows(),
              na_sensor.predict_obsrv(state, noise, 1.0).rows());
}

TEST(NotAdaptiveTests, Fixed_JointObsrvModel)
{
    using namespace fl;

    typedef LinearGaussianObservationModel<
                Eigen::Matrix<double, 2, 1>,
                Eigen::Matrix<double, 3, 1>
            > LocalObsrvModel;

    typedef JointObservationModel<MultipleOf<LocalObsrvModel, 3>> ObsrvModel;
    typedef NotAdaptive<ObsrvModel> NAObsrvModel;

    typedef typename Traits<ObsrvModel>::Obsrv Obsrv;
    typedef typename Traits<ObsrvModel>::State State;
    typedef typename Traits<ObsrvModel>::Noise Noise;
    typedef typename Traits<ObsrvModel>::Noise Noise;

    ObsrvModel sensor = ObsrvModel(LocalObsrvModel());
    NAObsrvModel na_sensor = NAObsrvModel(ObsrvModel(LocalObsrvModel()));

    State state = State::Random();
    Noise noise = Noise::Random();

    EXPECT_TRUE(
        fl::are_similar(
            sensor.predict_obsrv(state, noise, 1.0),
            na_sensor.predict_obsrv(state, noise, 1.0)));

    EXPECT_EQ(sensor.predict_obsrv(state, noise, 1.0).rows(),
              na_sensor.predict_obsrv(state, noise, 1.0).rows());
}
