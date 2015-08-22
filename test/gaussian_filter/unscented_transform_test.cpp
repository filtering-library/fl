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
 * \file robust_feature_obsrv_model.hpp
 * \date 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <fl/filter/gaussian/transform/point_set.hpp>
#include <fl/filter/gaussian/transform/unscented_transform.hpp>

TEST(UnscentedTransformTest, weights)
{
    typedef Eigen::Matrix<double, 10, 1> Point;
    typedef fl::PointSet<Point> Distribution;
    fl::UnscentedTransform ut;
    const double dim = 10;

    EXPECT_DOUBLE_EQ(1., ut.weight_mean_0(dim) + 2*dim * ut.weight_mean_i(dim));
}

template<
    template<typename, int> class PointSet,
    typename Point,
    int Dim
>
void test_mean_transform(PointSet<Point, Dim>& point_set, int dim)
{
    fl::UnscentedTransform ut;

    Point a = Point::Ones(dim) * 9;
    fl::Gaussian<Point> gaussian;

    gaussian.dimension(dim);
    gaussian.mean(a);

    //EXPECT_FALSE(a.isApprox(point_set.mean()));
    ut.forward(gaussian, point_set);
    EXPECT_TRUE(fl::are_similar(a, point_set.mean()));
}

// fixed dimension, fixed number of points
TEST(UnscentedTransformTest, mean_recovery_fixed_fixed)
{
    constexpr size_t dim = 10;
    typedef Eigen::Matrix<double, dim, 1> Point;

    fl::PointSet<
        Point,
        fl::UnscentedTransform::number_of_points(dim)
    > point_set;
    test_mean_transform(point_set, dim);
}

// fixed dimension, dynamic number of points
TEST(UnscentedTransformTest, mean_recovery_fixed_dynamic)
{
    constexpr size_t dim = 10;
    typedef Eigen::Matrix<double, dim, 1> Point;

    fl::PointSet<Point> point_set;
    test_mean_transform(point_set, dim);
}

// dynamic dimension, dynamic number of points
TEST(UnscentedTransformTest, mean_recovery_dynamic_dynamic)
{
    constexpr size_t dim = 10;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Point;
    typedef fl::PointSet<Point> PointSet;

    PointSet point_set(dim);
    test_mean_transform(point_set, dim);

    EXPECT_EQ(point_set.count_points(),
              fl::UnscentedTransform::number_of_points(dim));
}

// dynamic dimension, fixed number of points
TEST(UnscentedTransformTest, mean_recovery_dynamic_fixed_throw)
{
    constexpr size_t dim = 10;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Point;

    fl::UnscentedTransform ut;
    Point a = Point::Ones(dim) * 9;

    fl::Gaussian<Point> gaussian;
    gaussian.dimension(dim);
    gaussian.mean(a);

    fl::PointSet<Point, 3> point_set;

    EXPECT_THROW(ut.forward(gaussian, point_set),
                 fl::WrongSizeException);
}

// dynamic dimension, fixed number of points
TEST(UnscentedTransformTest, mean_recovery_dynamic_fixed)
{
    constexpr size_t dim = 10;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Point;
    typedef fl::PointSet<Point, 21> PointSet;

    PointSet point_set(dim);
    test_mean_transform(point_set, dim);

    EXPECT_EQ(point_set.count_points(),
              fl::UnscentedTransform::number_of_points(dim));
}


template<
    template<typename, int> class PointSet,
    typename Point,
    int Dim
>
void test_covariance_transform(PointSet<Point, Dim>& point_set, int dim)
{
    fl::UnscentedTransform ut;

    typename fl::Gaussian<Point>::SecondMoment cov;
    cov.setRandom(dim, dim);
    cov *= cov.transpose();

    Point a = Point::Random(dim);

    fl::Gaussian<Point> gaussian;
    gaussian.dimension(dim);
    gaussian.mean(a);
    gaussian.covariance(cov);

    EXPECT_NO_THROW(ut.forward(gaussian, point_set));

    EXPECT_TRUE(
        fl::are_similar(
            ( point_set.centered_points() *
              point_set.covariance_weights_vector().asDiagonal() *
              point_set.centered_points().transpose() ),
            cov));
}

TEST(UnscentedTransformTest, covariance_recovery_fixed_fixed)
{
    constexpr size_t dim = 10;
    typedef Eigen::Matrix<double, dim, 1> Point;

    fl::PointSet<
        Point,
        fl::UnscentedTransform::number_of_points(dim)
    > point_set;

    test_covariance_transform(point_set, dim);
}

TEST(UnscentedTransformTest, covariance_recovery_fixed_dynamic)
{
    constexpr size_t dim = 10;
    typedef Eigen::Matrix<double, dim, 1> Point;

    fl::PointSet<Point> point_set;
    test_covariance_transform(point_set, dim);
}

TEST(UnscentedTransformTest, covariance_recovery_dynamic_dynamic)
{
    constexpr size_t dim = 10;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Point;

    fl::PointSet<Point> point_set(dim);
    test_covariance_transform(point_set, dim);
}

TEST(UnscentedTransformTest, covariance_recovery_dynamic_fixed_throw)
{
    constexpr size_t dim = 10;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Point;

    fl::UnscentedTransform ut;

    typename fl::Gaussian<Point>::SecondMoment cov;
    cov.setRandom(dim, dim);
    cov *= cov.transpose();

    Point a = Point::Random(dim);

    fl::Gaussian<Point> gaussian;
    gaussian.dimension(dim);
    gaussian.mean(a);
    gaussian.covariance(cov);

    fl::PointSet<Point, 20> point_set;

    EXPECT_THROW(ut.forward(gaussian, point_set),
                 fl::WrongSizeException);
}

TEST(UnscentedTransformTest, covariance_recovery_dynamic_fixed)
{
    constexpr size_t dim = 10;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Point;

    fl::PointSet<Point, 21> point_set;
    point_set.dimension(dim);
    test_covariance_transform(point_set, dim);
}

