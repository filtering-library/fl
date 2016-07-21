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
 * \file robust_gaussian_filter_test.cpp
 * \date August 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "../typecast.hpp"
#include "gaussian_filter_test_suite.hpp"
#include <fl/util/meta.hpp>
#include <fl/filter/gaussian/gaussian_filter.hpp>
#include <fl/filter/gaussian/robust_gaussian_filter.hpp>
#include <fl/model/sensor/linear_cauchy_sensor.hpp>
#include <fl/model/sensor/body_tail_sensor.hpp>
#include <fl/model/sensor/linear_gaussian_sensor.hpp>

using namespace fl;

template <
    int StateDimension,
    int InputDimension,
    int ObsrvDimension,
    int FilterIterations
>
struct RobutGaussianFilterTestConfiguration
{
    enum : signed int
    {
        StateDim = StateDimension,
        InputDim = InputDimension,
        ObsrvDim = ObsrvDimension,
        Iterations = FilterIterations
    };

    template <typename ModelFactory>
    struct FilterDefinition
    {
        typedef typename ModelFactory::LinearObservation::Obsrv Obsrv;
        typedef typename ModelFactory::LinearObservation::State State;

        typedef fl::LinearCauchySensor<Obsrv, State> CauchyModel;

        typedef fl::BodyTailSensor<
                    typename ModelFactory::LinearObservation,
                    CauchyModel
                > BodyTailSensor;

        typedef UnscentedQuadrature Quadrature;
//        typedef fl::SigmaPointQuadrature<
//                    fl::MonteCarloTransform<
//                        fl::ConstantPointCountPolicy<1000>>> Quadrature;

        typedef RobustGaussianFilter<
                    typename ModelFactory::LinearTransition,
                    BodyTailSensor,
                    Quadrature
                > Type;
    };

    template <typename ModelFactory>
    static typename FilterDefinition<ModelFactory>::Type
    create_filter(ModelFactory&& factory)
    {
        typedef FilterDefinition<ModelFactory> Definition;
        typedef typename Definition::Type Filter;
        typedef typename Definition::CauchyModel CauchyModel;
        typedef typename Definition::BodyTailSensor BodyTailSensor;

        auto body_model = factory.create_sensor();
        auto tail_model = CauchyModel();
        tail_model.noise_covariance(tail_model.noise_covariance() * 10.);

        return Filter(
            factory.create_linear_state_model(),
            BodyTailSensor(body_model, tail_model, 0.1),
            typename Definition::Quadrature());
    }
};


typedef ::testing::Types<
            StaticTest<RobutGaussianFilterTestConfiguration<1, 1, 1, 30>>
        > TestTypes;

INSTANTIATE_TYPED_TEST_CASE_P(RobustGaussianFilterTest,
                              GaussianFilterTest,
                              TestTypes);
