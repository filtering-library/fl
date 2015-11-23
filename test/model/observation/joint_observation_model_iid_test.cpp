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
 * \file joint_observation_model_iid_test.cpp
 * \date August 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>
#include "../../typecast.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <iostream>

#include <fl/util/types.hpp>
#include <fl/model/observation/linear_gaussian_observation_model.hpp>
#include <fl/model/observation/joint_observation_model_iid.hpp>

template <typename TestType>
class JointObservationModelIidTest:
    public testing::Test
{
public:
    enum : signed int
    {
        LocalStateDim = TestType::Parameter::StateDim,
        LocalObsrvDim = TestType::Parameter::ObsrvDim,
        LocalNoiseDim = TestType::Parameter::ObsrvDim,
        Count = TestType::Parameter::LocalModelCount,

        LocalStateSize = fl::TestSize<LocalStateDim, TestType>::Value,
        LocalObsrvSize = fl::TestSize<LocalObsrvDim, TestType>::Value,
        LocalNoiseSize = fl::TestSize<LocalNoiseDim, TestType>::Value,
        Size = fl::TestSize<Count, TestType>::Value
    };

    // Local model types
    typedef Eigen::Matrix<fl::Real, LocalStateSize, 1> LocalState;
    typedef Eigen::Matrix<fl::Real, LocalObsrvSize, 1> LocalObsrv;
    typedef fl::LinearGaussianObservationModel<LocalObsrv, LocalState> LocalModel;
    typedef typename LocalModel::Noise LocalNoise;

    // Joint model types
    typedef fl::JointObservationModel<
                fl::MultipleOf<LocalModel, Size>
            > JointModel;

    typedef typename JointModel::Obsrv Obsrv;
    typedef typename JointModel::Noise Noise;
    typedef typename JointModel::State State;

    enum : signed int
    {
        StateDim = LocalStateDim,
        ObsrvDim = LocalObsrvDim * Count,
        NoiseDim = LocalNoiseDim * Count,

        StateSize = fl::TestSize<StateDim, TestType>::Value,
        ObsrvSize = fl::TestSize<ObsrvDim, TestType>::Value,
        NoiseSize = fl::TestSize<NoiseDim, TestType>::Value
    };

    JointObservationModelIidTest()
        : joint_model(create_joint_observation_model(create_local_model())),
          noise_normal_gaussian(NoiseDim)
    { }

    State zero_state() { return State::Zero(LocalStateDim); }
    Obsrv zero_obsrv() { return Obsrv::Zero(LocalObsrvDim); }
    Noise zero_noise() { return Noise::Zero(LocalNoiseDim); }

    State rand_state() { return State::Random(LocalStateDim); }
    Obsrv rand_obsrv() { return Obsrv::Random(LocalObsrvDim); }
    Noise rand_noise() { return noise_normal_gaussian.sample(); }

    LocalModel create_local_model()
    {
        return LocalModel(LocalObsrvDim, LocalStateDim);
    }

    template <typename LocalObsrvModel>
    JointModel create_joint_observation_model(LocalObsrvModel&& local_model)
    {
        return JointModel(local_model, Count);
    }

    void init_dimension_test()
    {
        EXPECT_EQ(joint_model.obsrv_dimension(), ObsrvDim);
        EXPECT_EQ(joint_model.noise_dimension(), NoiseDim);
        EXPECT_EQ(joint_model.state_dimension(), StateDim);
    }

    void observation()
    {
        auto y = joint_model.observation(rand_state(), rand_noise());

        EXPECT_EQ(y.size(), ObsrvDim);
        for (int i = 0; i < y.size(); ++i)
        {
            EXPECT_NE(y[i], 0.);
        }
    }

    void nested_joint_model()
    {
        // nested joint model type taking multiple of JointModel
        typedef fl::JointObservationModel<
                    fl::MultipleOf<JointModel, Size>
                >  OuterJointModel;

        auto nested_joint =
            OuterJointModel(
                create_joint_observation_model(create_local_model()), Count);

        EXPECT_EQ(nested_joint.obsrv_dimension(), ObsrvDim * Count);
        EXPECT_EQ(nested_joint.noise_dimension(), NoiseDim * Count);
        EXPECT_EQ(nested_joint.state_dimension(), StateDim);
    }

//    void probability()
//    {
//        auto x = State::Random(StateDim).eval();
//        auto y = Obsrv::Random(ObsrvDim).eval();

//        auto body_prob = body_model.probability(y, x);
//        auto tail_prob = tail_model.probability(y, x);
//        auto body_tail_prob = body_tail_model.probability(y, x);

//        ASSERT_DOUBLE_EQ((body_prob + tail_prob)/2., body_tail_prob);
//    }

//    void log_probability()
//    {
//        auto x = State::Random(StateDim).eval();
//        auto y = Obsrv::Random(ObsrvDim).eval();

//        auto body_prob = body_model.probability(y, x);
//        auto tail_prob = tail_model.probability(y, x);
//        auto body_tail_log_prob = body_tail_model.log_probability(y, x);

//        ASSERT_DOUBLE_EQ(std::log((body_prob + tail_prob)/2.),
//                         body_tail_log_prob);
//    }

protected:
    JointModel joint_model;
    fl::StandardGaussian<Noise> noise_normal_gaussian;
};

template <int ModelCount, int ObsrvDimension, int StateDimension>
struct Dimensions
{
    enum: signed int
    {
        ObsrvDim = ObsrvDimension,
        StateDim = StateDimension,
        LocalModelCount = ModelCount
    };
};

typedef ::testing::Types<
            fl::StaticTest<Dimensions<1, 1, 1>>,
            fl::StaticTest<Dimensions<2, 1, 1>>,
            fl::StaticTest<Dimensions<10, 1, 1>>,
            fl::StaticTest<Dimensions<1, 2, 1>>,
            fl::StaticTest<Dimensions<2, 3, 2>>,
            fl::StaticTest<Dimensions<10, 11, 1>>,
            fl::DynamicTest<Dimensions<1, 1, 1>>,
            fl::DynamicTest<Dimensions<2, 1, 1>>,
            fl::DynamicTest<Dimensions<10, 1, 1>>,
            fl::DynamicTest<Dimensions<1, 2, 1>>,
            fl::DynamicTest<Dimensions<2, 3, 2>>,
            fl::DynamicTest<Dimensions<10, 11, 1>>
        > TestTypes;

TYPED_TEST_CASE(JointObservationModelIidTest, TestTypes);

TYPED_TEST(JointObservationModelIidTest, init_dimension)
{
    TestFixture::init_dimension_test();
}

TYPED_TEST(JointObservationModelIidTest, observation)
{
    TestFixture::observation();
}

TYPED_TEST(JointObservationModelIidTest, nested_joint_model)
{
    TestFixture::nested_joint_model();
}
