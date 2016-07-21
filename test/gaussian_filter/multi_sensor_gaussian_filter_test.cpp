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
 * \file multi_sensor_gaussian_filter_test.cpp
 * \date August 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>
#include "../typecast.hpp"

#include <Eigen/Dense>

#include "gaussian_filter_test_suite.hpp"
#include <fl/util/meta.hpp>
#include <fl/filter/gaussian/quadrature/unscented_quadrature.hpp>
#include <fl/filter/gaussian/multi_sensor_gaussian_filter.hpp>

using namespace fl;

template <
    int StateDimension,
    int InputDimension,
    int ObsrvDimension,
    int Count,               // Local observation model count
    int FilterIterations
>
struct MultiSensorGfTestConfiguration
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
        enum : signed int
        {
            // compile time size (positive for static and -1 for dynamic)
            Size  = ExpandSizes<Count, ModelFactory::Sizes>::Value
        };

        typedef typename ModelFactory::LinearObservation LocalSensor;
        typedef JointSensor<
                    MultipleOf<LocalSensor, Size>
                > JointSensor;

        typedef UnscentedQuadrature Quadrature;

        typedef MultiSensorGaussianFilter<
                        typename ModelFactory::LinearTransition,
                        JointSensor,
                        Quadrature
                > Type;
    };

    template <typename ModelFactory>
    static typename FilterDefinition<ModelFactory>::Type
    create_filter(ModelFactory&& factory)
    {
        typedef typename
        FilterDefinition<ModelFactory>::JointSensor JointSensor;

        return typename FilterDefinition<ModelFactory>::Type(
            factory.create_linear_state_model(),
            JointSensor(factory.create_observation_model(), Count),
            UnscentedQuadrature());
    }
};

typedef ::testing::Types<
            StaticTest<MultiSensorGfTestConfiguration<6, 1, 2, 10, 10>>
        > TestTypes;

INSTANTIATE_TYPED_TEST_CASE_P(MultiSensorGaussianFilterTest,
                              GaussianFilterTest,
                              TestTypes);

