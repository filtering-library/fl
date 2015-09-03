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
#include <fl/filter/gaussian/robust_multi_sensor_gaussian_filter.hpp>
#include <fl/model/observation/linear_cauchy_observation_model.hpp>
#include <fl/model/observation/body_tail_observation_model.hpp>
#include <fl/model/observation/linear_gaussian_observation_model.hpp>

using namespace fl;

template <
    int StateDimension,
    int ObsrvDimension,
    int Count,               // Local observation model count
    int FilterIterations
>
struct RobustMultiSensorGfTestConfiguration
{
    enum : signed int
    {
        StateDim = StateDimension,
        InputDim = 1,
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

        // ================================================================== //
        // == Define Process Model                                         == //
        // ================================================================== //
        typedef typename ModelFactory::LinearStateTransition ProcessModel;

        // ================================================================== //
        // == Define Body Tail Observation Model                           == //
        // ================================================================== //
        typedef typename ModelFactory::LinearObservation::Obsrv Obsrv;
        typedef typename ModelFactory::LinearObservation::State State;

        typedef fl::LinearCauchyObservationModel<Obsrv, State> CauchyModel;

        typedef fl::BodyTailObsrvModel<
                    typename ModelFactory::LinearObservation,
                    CauchyModel
                > BodyTailObsrvModel;

        typedef BodyTailObsrvModel LocalObsrvModel;

        // ================================================================== //
        // == Define Joint Body Tail Observation Model                     == //
        // ================================================================== //
        typedef JointObservationModel<
                    MultipleOf<LocalObsrvModel, Size>
                > JointObsrvModel;

        // ================================================================== //
        // == Define Integration Quadrature                                == //
        // ================================================================== //
        typedef UnscentedQuadrature Quadrature;

        // ================================================================== //
        // == Define the filter                                            == //
        // ================================================================== //
        typedef RobustMultiSensorGaussianFilter<
                    ProcessModel, JointObsrvModel, Quadrature
                > Type;
    };

    template <typename ModelFactory>
    static typename FilterDefinition<ModelFactory>::Type
    create_filter(ModelFactory&& factory)
    {
        typedef FilterDefinition<ModelFactory> Definition;

        typedef typename Definition::Type Filter;
        typedef typename Definition::CauchyModel CauchyModel;
        typedef typename Definition::BodyTailObsrvModel BodyTailObsrvModel;
        typedef typename Definition::JointObsrvModel JointObsrvModel;

        auto body_model = factory.create_observation_model();
        auto tail_model = CauchyModel();
        tail_model.noise_covariance(tail_model.noise_covariance() * 10.);

        return Filter(
            factory.create_linear_state_model(),
            JointObsrvModel(BodyTailObsrvModel(body_model, tail_model, 0.1), Count),
            typename Definition::Quadrature());
    }
};

typedef ::testing::Types<
            StaticTest<RobustMultiSensorGfTestConfiguration<12, 1, 1200, 30>>
        > TestTypes;

INSTANTIATE_TYPED_TEST_CASE_P(RobustMultiSensorGaussianFilterTest,
                              GaussianFilterTest,
                              TestTypes);
