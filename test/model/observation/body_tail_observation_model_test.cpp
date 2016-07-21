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
 * \file body_tail_observation_model_test.hpp
 * \date August 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>
#include "../../typecast.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <iostream>

#include <fl/util/types.hpp>
#include <fl/model/observation/body_tail_observation_model.hpp>
#include <fl/model/observation/linear_gaussian_observation_model.hpp>
#include <fl/model/observation/linear_decorrelated_gaussian_observation_model.hpp>

template <typename TestType>
class BodyTailSensorTest:
    public testing::Test
{
public:
    enum : signed int
    {
        StateDim = TestType::Parameter::StateDim,
        ObsrvDim = TestType::Parameter::ObsrvDim,
        NoiseDim = TestType::Parameter::ObsrvDim + 1,

        StateSize = fl::TestSize<StateDim, TestType>::Value,
        ObsrvSize = fl::TestSize<ObsrvDim, TestType>::Value,
        NoiseSize = fl::TestSize<NoiseDim, TestType>::Value
    };

    typedef Eigen::Matrix<fl::Real, StateSize, 1> State;
    typedef Eigen::Matrix<fl::Real, ObsrvSize, 1> Obsrv;

    typedef fl::LinearGaussianSensor<Obsrv, State> BodyModel;
    typedef fl::LinearGaussianSensor<Obsrv, State> TailModel;
    typedef fl::BodyTailSensor<BodyModel, TailModel> BodyTailModel;

    typedef typename BodyTailModel::Noise Noise;

    BodyTailSensorTest()
        : body_model(create_body_model()),
          tail_model(create_tail_model()),
          body_tail_model(create_body_tail_model(body_model, tail_model))
    { }

    BodyModel create_body_model()
    {
        auto model = BodyModel(ObsrvDim, StateDim);

        auto noise_matrix = model.create_noise_matrix();
        noise_matrix *= fl::Real(10.0);
        model.noise_matrix(noise_matrix);

        return model;
    }

    TailModel create_tail_model()
    {
        auto model = TailModel(ObsrvDim, StateDim);

        auto noise_matrix = model.create_noise_matrix();
        noise_matrix *= fl::Real(0.1);
        model.noise_matrix(noise_matrix);

        return model;
    }

    BodyTailModel create_body_tail_model(const BodyModel& body_model,
                                         const TailModel& tail_model,
                                         fl::Real threshold = 0.5)
    {
        auto model = BodyTailModel(body_model, tail_model, threshold);

        return model;
    }

    void init_dimension_test()
    {
        EXPECT_EQ(body_tail_model.obsrv_dimension(), ObsrvDim);
        EXPECT_EQ(body_tail_model.noise_dimension(), ObsrvDim + 1);
        EXPECT_EQ(body_tail_model.state_dimension(), StateDim);

        EXPECT_EQ(body_tail_model.obsrv_dimension(),
                  body_model.obsrv_dimension());
        EXPECT_EQ(body_tail_model.state_dimension(),
                  body_model.state_dimension());

        EXPECT_EQ(body_tail_model.obsrv_dimension(),
                  tail_model.obsrv_dimension());
        EXPECT_EQ(body_tail_model.state_dimension(),
                  tail_model.state_dimension());

        EXPECT_EQ(body_tail_model.noise_dimension(),
                  1 + std::max(body_model.noise_dimension(),
                               tail_model.noise_dimension()));
    }

    void observation(fl::Real threshold)
    {
        auto x = State::Random(StateDim).eval();
        auto n = Noise::Random(NoiseDim).eval();

        n(body_tail_model.noise_dimension() - 1)
            = fl::uniform_to_normal(threshold);

        auto body_y = body_model.observation(
            x, n.topRows(body_model.noise_dimension()));
        auto tail_y = tail_model.observation(
            x, n.topRows(tail_model.noise_dimension()));
        auto body_tail_y = body_tail_model.observation(x, n);

        if (threshold > 0.5)
        {
            EXPECT_TRUE(fl::are_similar(body_y, body_tail_y));
            EXPECT_FALSE(fl::are_similar(tail_y, body_tail_y));
        }
        else
        {
            EXPECT_FALSE(fl::are_similar(body_y, body_tail_y));
            EXPECT_TRUE(fl::are_similar(tail_y, body_tail_y));
        }
    }

    void probability()
    {
        auto x = State::Random(StateDim).eval();
        auto y = Obsrv::Random(ObsrvDim).eval();

        auto body_prob = body_model.probability(y, x);
        auto tail_prob = tail_model.probability(y, x);
        auto body_tail_prob = body_tail_model.probability(y, x);

        ASSERT_DOUBLE_EQ((body_prob + tail_prob)/2., body_tail_prob);
    }

    void log_probability()
    {
        auto x = State::Random(StateDim).eval();
        auto y = Obsrv::Random(ObsrvDim).eval();

        auto body_prob = body_model.probability(y, x);
        auto tail_prob = tail_model.probability(y, x);
        auto body_tail_log_prob = body_tail_model.log_probability(y, x);

        ASSERT_DOUBLE_EQ(std::log((body_prob + tail_prob)/2.),
                         body_tail_log_prob);
    }

protected:
    BodyModel body_model;
    TailModel tail_model;
    BodyTailModel body_tail_model;
};

template <int ObsrvDimension, int StateDimension>
struct Dimensions
{
    enum: signed int
    {
        ObsrvDim = ObsrvDimension,
        StateDim = StateDimension
    };
};

typedef ::testing::Types<
            fl::StaticTest<Dimensions<2, 1>>,
            fl::StaticTest<Dimensions<2, 2>>,
            fl::StaticTest<Dimensions<3, 3>>,
            fl::StaticTest<Dimensions<10, 10>>,
            fl::DynamicTest<Dimensions<2, 1>>,
            fl::DynamicTest<Dimensions<2, 2>>,
            fl::DynamicTest<Dimensions<3, 3>>,
            fl::DynamicTest<Dimensions<10, 10>>
        > TestTypes;

TYPED_TEST_CASE(BodyTailSensorTest, TestTypes);

TYPED_TEST(BodyTailSensorTest, init_dimension)
{
    TestFixture::init_dimension_test();
}

TYPED_TEST(BodyTailSensorTest, observation_select_body)
{
    TestFixture::observation(0.6);
}

TYPED_TEST(BodyTailSensorTest, observation_select_tail)
{
    TestFixture::observation(0.4);
}

TYPED_TEST(BodyTailSensorTest, observation_select_randomly)
{
    for (int i = 0; i < 1000; ++i)
    {
        TestFixture::observation(fl::Real(rand()) / fl::Real(RAND_MAX));
    }
}

TYPED_TEST(BodyTailSensorTest, probability)
{
    for (int i = 0; i < 1000; ++i)
    {
        TestFixture::probability();
    }
}

TYPED_TEST(BodyTailSensorTest, log_probability)
{
    for (int i = 0; i < 1000; ++i)
    {
        TestFixture::log_probability();
    }
}

TYPED_TEST(BodyTailSensorTest, wrong_threshlold_exception)
{
    EXPECT_THROW(TestFixture::create_body_tail_model(
                     TestFixture::create_body_model(),
                     TestFixture::create_tail_model(),
                     fl::Real(-0.1)),
                 fl::Exception);

    EXPECT_THROW(TestFixture::create_body_tail_model(
                     TestFixture::create_body_model(),
                     TestFixture::create_tail_model(),
                     fl::Real(1.1)),
                 fl::Exception);
}

TYPED_TEST(BodyTailSensorTest, correct_threshlold)
{
    EXPECT_NO_THROW(TestFixture::create_body_tail_model(
                     TestFixture::create_body_model(),
                     TestFixture::create_tail_model(),
                     fl::Real(0.99999999)));

    EXPECT_NO_THROW(TestFixture::create_body_tail_model(
                     TestFixture::create_body_model(),
                     TestFixture::create_tail_model(),
                     fl::Real(0.000000001)));
}
