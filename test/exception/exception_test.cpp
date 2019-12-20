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
 * \date 2014
 * \author Jan Issac (jan.issac@gmail.com)
 * Max-Planck-Institute for Intelligent Systems, University of Southern California
 */

#include <gtest/gtest.h>

#include <fl/exception/exception.hpp>
#include <fl/filter/gaussian/transform/point_set.hpp>

TEST(Exception, create)
{
    struct FirstWeight
    {
        double w;
        std::string name;
    };

    typedef fl::PointSet<Eigen::Matrix<double, 1, 1>, -1> SigmaPointGaussian;
    SigmaPointGaussian sigmas(1);

    try
    {
        sigmas.point(0, Eigen::Matrix<double, 1, 1>::Random(), {1.23, 1.24});
        sigmas.point(1, Eigen::Matrix<double, 1, 1>::Random(), {1.23, 1.24});
        sigmas.point(2, Eigen::Matrix<double, 1, 1>::Random(), {1.23, 1.24});
    }
    catch(fl::OutOfBoundsException& e) { }
    catch(...)
    {
        ADD_FAILURE();
    }
}
