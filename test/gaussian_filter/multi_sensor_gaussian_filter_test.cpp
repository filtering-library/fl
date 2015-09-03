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

        typedef typename ModelFactory::LinearObservation LocalObsrvModel;
        typedef JointObservationModel<
                    MultipleOf<LocalObsrvModel, Size>
                > JointObsrvModel;

        typedef UnscentedQuadrature Quadrature;

        typedef MultiSensorGaussianFilter<
                        typename ModelFactory::LinearStateTransition,
                        JointObsrvModel,
                        Quadrature
                > Type;
    };

    template <typename ModelFactory>
    static typename FilterDefinition<ModelFactory>::Type
    create_filter(ModelFactory&& factory)
    {
        typedef typename
        FilterDefinition<ModelFactory>::JointObsrvModel JointObsrvModel;

        return typename FilterDefinition<ModelFactory>::Type(
            factory.create_linear_state_model(),
            JointObsrvModel(factory.create_observation_model(), Count),
            UnscentedQuadrature());
    }
};

typedef ::testing::Types<
            StaticTest<MultiSensorGfTestConfiguration<12, 1, 3, 10, 100>>
        > TestTypes;

INSTANTIATE_TYPED_TEST_CASE_P(MultiSensorGaussianFilterTest,
                              GaussianFilterTest,
                              TestTypes);
