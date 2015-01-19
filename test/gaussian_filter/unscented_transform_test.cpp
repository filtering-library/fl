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
 * @date 2014
 * @author Jan Issac (jan.issac@gmail.com)
 * Max-Planck-Institute for Intelligent Systems,
 * University of Southern California
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <fl/filter/gaussian/point_set.hpp>
#include <fl/filter/gaussian/unscented_transform.hpp>

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
    EXPECT_TRUE(a.isApprox(point_set.mean()));
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

    typename fl::Traits<fl::Gaussian<Point>>::Operator cov;
    cov.setRandom(dim, dim);
    cov *= cov.transpose();

    Point a = Point::Random(dim);

    fl::Gaussian<Point> gaussian;
    gaussian.dimension(dim);
    gaussian.mean(a);
    gaussian.covariance(cov);

    EXPECT_NO_THROW(ut.forward(gaussian, point_set));

    EXPECT_TRUE((point_set.centered_points() *
                 point_set.covariance_weights_vector().asDiagonal() *
                 point_set.centered_points().transpose())
                .isApprox(cov));
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

    typename fl::Traits<fl::Gaussian<Point>>::Operator cov;
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

