/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Max-Planck-Institute for Intelligent Systems,
 *                     University of Southern California
 *    Jan Issac (jan.issac@gmail.com)
 *    Manuel Wuthrich (manuel.wuthrich@gmail.com)
 *
 *
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

/**
 * \date 2014
 * \author Jan Issac (jan.issac@gmail.com)
 * Max-Planck-Institute for Intelligent Systems,
 * University of Southern California
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <iostream>

#include <fl/model/observation/linear_observation_model.hpp>

class LinearObservationModelTests:
        public testing::Test
{
public:
    template <typename Model>
    void InitDimensionTests(Model& model,
                            size_t dim,
                            size_t dim_state,
                            typename Model::SecondMoment& cov)
    {
        EXPECT_EQ(model.obsrv_dimension(), dim);
        EXPECT_EQ(model.standard_variate_dimension(), dim);
        EXPECT_EQ(model.state_dimension(), dim_state);
        EXPECT_TRUE(model.H().isOnes());
        EXPECT_TRUE(model.covariance().isApprox(cov));
    }
};

TEST_F(LinearObservationModelTests, init_fixedsize_dimension)
{
    typedef Eigen::Matrix<double, 10, 1> State;
    typedef Eigen::Matrix<double, 20, 1> Obsrv;
    const size_t dim = Obsrv::SizeAtCompileTime;
    const size_t dim_state = State::SizeAtCompileTime;
    typedef fl::LinearGaussianObservationModel<Obsrv, State> LGModel;

    LGModel::SecondMoment cov = LGModel::SecondMoment::Identity() * 5.5465;
    LGModel model(cov);

    InitDimensionTests(model, dim, dim_state, cov);
}

TEST_F(LinearObservationModelTests, init_dynamicsize_dimension)
{
    const size_t dim = 20;
    const size_t dim_state = 10;
    typedef Eigen::VectorXd State;
    typedef Eigen::VectorXd Obsrv;
    typedef fl::LinearGaussianObservationModel<Obsrv, State> LGModel;

    LGModel::SecondMoment cov = LGModel::SecondMoment::Identity(dim, dim) * 5.5465;
    LGModel model(cov, dim, dim_state);

    InitDimensionTests(model, dim, dim_state, cov);
}

TEST_F(LinearObservationModelTests, predict_fixedsize_with_zero_noise)
{
    typedef Eigen::Matrix<double, 10, 1> State;
    typedef Eigen::Matrix<double, 20, 1> Obsrv;
    const size_t dim = Obsrv::SizeAtCompileTime;
    const size_t dim_state = State::SizeAtCompileTime;
    typedef fl::LinearGaussianObservationModel<Obsrv, State> LGModel;

    State state = State::Random(dim_state, 1);
    Obsrv observation = Obsrv::Random(dim, 1);
    LGModel::Noise noise = LGModel::Noise::Zero(dim, 1);
    LGModel::SecondMoment cov = LGModel::SecondMoment::Identity(dim, dim) * 5.5465;
    LGModel model(cov);

    EXPECT_TRUE(model.map_standard_normal(noise).isZero());

    EXPECT_FALSE(model.map_standard_normal(noise).isApprox(observation));
    model.condition(state);
    EXPECT_FALSE(model.map_standard_normal(noise).isApprox(observation));
}

TEST_F(LinearObservationModelTests, predict_dynamic_with_zero_noise)
{
    const size_t dim = 20;
    const size_t dim_state = 10;
    typedef Eigen::VectorXd State;
    typedef Eigen::VectorXd Obsrv;
    typedef fl::LinearGaussianObservationModel<Obsrv, State> LGModel;

    State state = State::Random(dim_state, 1);
    Obsrv observation = Obsrv::Random(dim, 1);
    LGModel::Noise noise = LGModel::Noise::Zero(dim, 1);
    LGModel::SecondMoment cov = LGModel::SecondMoment::Identity(dim, dim) * 5.5465;
    LGModel model(cov, dim, dim_state);

    EXPECT_TRUE(model.map_standard_normal(noise).isZero());

    EXPECT_FALSE(model.map_standard_normal(noise).isApprox(observation));
    model.condition(state);
    EXPECT_FALSE(model.map_standard_normal(noise).isApprox(observation));
}

TEST_F(LinearObservationModelTests, sensor_matrix)
{
    const size_t dim = 20;
    const size_t dim_state = 10;
    typedef Eigen::VectorXd State;
    typedef Eigen::VectorXd Obsrv;
    typedef fl::LinearGaussianObservationModel<Obsrv, State> LGModel;

    State state = State::Random(dim_state, 1);
    Obsrv observation = Obsrv::Zero(dim, 1);
    LGModel::Noise noise = LGModel::Noise::Random(dim, 1);
    LGModel::SecondMoment cov = LGModel::SecondMoment::Identity(dim, dim);
    LGModel::SensorMatrix H = LGModel::SecondMoment::Ones(dim, dim_state);
    LGModel model(cov, dim, dim_state);

    observation.topRows(dim_state) = state;

    EXPECT_TRUE(model.map_standard_normal(noise).isApprox(noise));
    EXPECT_FALSE(model.map_standard_normal(noise).isApprox(observation));

    model.condition(state);

    EXPECT_TRUE(model.map_standard_normal(noise).isApprox(H * state + noise));
    EXPECT_FALSE(model.map_standard_normal(noise).isApprox(observation));

    H = LGModel::SecondMoment::Zero(dim, dim_state);
    H.block(0, 0, dim_state, dim_state)
            = Eigen::MatrixXd::Identity(dim_state, dim_state);

    model.H(H);
    model.condition(state);

    EXPECT_TRUE(model.map_standard_normal(noise).isApprox(observation + noise));
}

