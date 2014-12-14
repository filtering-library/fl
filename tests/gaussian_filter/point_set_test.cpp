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

#include <memory.h>

#include <fl/exception/exception.hpp>
#include <fl/filter/gaussian/point_set.hpp>


TEST(PointSet, fixed_fixed)
{
    fl::PointSet<Eigen::Matrix<double, 10, 1>, 20> sigmas;

    EXPECT_NO_THROW(sigmas.point(0));
    EXPECT_NO_THROW(sigmas.point(19));
    EXPECT_THROW(sigmas.point(20), fl::OutOfBoundsException);
    // sigmas.resize(5); // doesn't compile as expected
    EXPECT_EQ(sigmas.count_points(), 20);
}

template <typename Distribution>
void test_fixed_dynamic_default(Distribution& sigmas)
{
    EXPECT_EQ(sigmas.count_points(), 0);
    EXPECT_THROW(sigmas.point(0), fl::OutOfBoundsException);
    EXPECT_THROW(sigmas.point(19), fl::OutOfBoundsException);
    sigmas.resize(5);
    EXPECT_NO_THROW(sigmas.point(0));
    EXPECT_NO_THROW(sigmas.point(4));
    EXPECT_THROW(sigmas.point(5), fl::OutOfBoundsException);
    EXPECT_EQ(sigmas.count_points(), 5);
}

template <typename Distribution>
void test_fixed_dynamic(Distribution& sigmas, size_t initial_size)
{
    EXPECT_THROW(sigmas.point(initial_size), fl::OutOfBoundsException);
    EXPECT_EQ(sigmas.count_points(), initial_size);
    sigmas.resize(5);
    EXPECT_NO_THROW(sigmas.point(0));
    EXPECT_NO_THROW(sigmas.point(4));
    EXPECT_THROW(sigmas.point(5), fl::OutOfBoundsException);
    EXPECT_EQ(sigmas.count_points(), 5);
}

TEST(PointSet, fixed_dynamic)
{
    constexpr size_t dim = 10;

    {SCOPED_TRACE("fixed_dynamic_default");
        fl::PointSet<Eigen::Matrix<double, dim, 1>> sigmas;
        test_fixed_dynamic_default(sigmas);
    }

    {SCOPED_TRACE("fixed_dynamic_0");
        fl::PointSet<Eigen::Matrix<double, dim, 1>> sigmas(dim, 0);
        test_fixed_dynamic_default(sigmas);
    }

    {SCOPED_TRACE("fixed_dynamic");
        fl::PointSet<Eigen::Matrix<double, 10, 1>> sigmas(dim, 4);
        test_fixed_dynamic(sigmas, 4);
    }
}

template <typename Distribution>
void test_dynamic_dynamic_default(Distribution& sigmas)
{
    EXPECT_EQ(sigmas.count_points(), 0);
    EXPECT_THROW(sigmas.point(0), fl::OutOfBoundsException);
    EXPECT_THROW(sigmas.point(19), fl::OutOfBoundsException);
    sigmas.resize(5);
    EXPECT_THROW(sigmas.point(0), fl::ZeroDimensionException);
    EXPECT_THROW(sigmas.point(4), fl::ZeroDimensionException);
    sigmas.dimension(5);
    EXPECT_NO_THROW(sigmas.point(0));
    EXPECT_NO_THROW(sigmas.point(4));
    EXPECT_THROW(sigmas.point(5), fl::OutOfBoundsException);
    EXPECT_EQ(sigmas.count_points(), 5);
}

template <typename Distribution>
void test_dynamic_dynamic(Distribution& sigmas, size_t initial_size)
{
    EXPECT_THROW(sigmas.point(initial_size), fl::OutOfBoundsException);
    EXPECT_EQ(sigmas.count_points(), initial_size);
    sigmas.resize(5);
    EXPECT_THROW(sigmas.point(0), fl::ZeroDimensionException);
    EXPECT_THROW(sigmas.point(4), fl::ZeroDimensionException);
    sigmas.dimension(5);
    EXPECT_NO_THROW(sigmas.point(0));
    EXPECT_NO_THROW(sigmas.point(4));
    EXPECT_THROW(sigmas.point(5), fl::OutOfBoundsException);
    EXPECT_EQ(sigmas.count_points(), 5);
}

TEST(PointSet, dynamic_dynamic)
{
    {SCOPED_TRACE("dynamic_dynamic_default");
        fl::PointSet<Eigen::Matrix<double, Eigen::Dynamic, 1>> sigmas;
        test_dynamic_dynamic_default(sigmas);
    }

    {SCOPED_TRACE("dynamic_dynamic_0");
        fl::PointSet<Eigen::Matrix<double, Eigen::Dynamic, 1>> sigmas(0, 0);
        test_dynamic_dynamic_default(sigmas);
    }

    {SCOPED_TRACE("dynamic_dynamic");
        fl::PointSet<Eigen::Matrix<double, Eigen::Dynamic, 1>> sigmas(0, 4);
        test_dynamic_dynamic(sigmas, 4);
    }
}

template <typename Distribution, typename Point>
void point_setter_tests(Distribution& sigmas, Point& p, size_t dimension)
{
    sigmas.point(0, p);
    EXPECT_TRUE(sigmas.point(0).isApprox(p));
    EXPECT_FALSE(sigmas.point(1).isApprox(p));
    EXPECT_FALSE(sigmas.point(2).isApprox(p));
    EXPECT_FALSE(sigmas.point(3).isApprox(p));

    p.setRandom();
    sigmas.point(1, p, 0.);
    EXPECT_FALSE(sigmas.point(0).isApprox(p));
    EXPECT_TRUE(sigmas.point(1).isApprox(p));
    EXPECT_FALSE(sigmas.point(2).isApprox(p));
    EXPECT_FALSE(sigmas.point(3).isApprox(p));

    p.setRandom();
    sigmas.point(2, p, 0., 0.);
    EXPECT_FALSE(sigmas.point(0).isApprox(p));
    EXPECT_FALSE(sigmas.point(1).isApprox(p));
    EXPECT_TRUE(sigmas.point(2).isApprox(p));
    EXPECT_FALSE(sigmas.point(3).isApprox(p));

    p.setRandom();
    sigmas.point(3, p, typename fl::Traits<Distribution>::Weight{0., 0.});
    EXPECT_FALSE(sigmas.point(0).isApprox(p));
    EXPECT_FALSE(sigmas.point(1).isApprox(p));
    EXPECT_FALSE(sigmas.point(2).isApprox(p));
    EXPECT_TRUE(sigmas.point(3).isApprox(p));
}

TEST(PointSet, point_setter_fixed)
{
    constexpr size_t dim = 5;

    typedef Eigen::Matrix<double, dim, 1> Point ;

    fl::PointSet<Point> sigmas(dim, 2 * Point::SizeAtCompileTime + 1);

    Point p = Point::Random();
    point_setter_tests(sigmas, p, dim);
}

TEST(PointSet, point_setter_dynamic)
{
    constexpr size_t dim = 5;

    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Point ;

    fl::PointSet<Point> sigmas(dim, 2 * dim + 1);

    Point p = Point::Random(dim);
    point_setter_tests(sigmas, p, dim);
}


template <typename Distribution>
void point_weight_setter_tests(Distribution& sigmas)
{
    typename fl::Traits<Distribution>::Point p(5);

    const double w0 = 0.554;
    const double w1_mean = 0.6425434;
    const double w1_cov = 0.7325;

    sigmas.point(1, p, w0);
    EXPECT_NE(sigmas.weights(0).w_mean, w0);
    EXPECT_NE(sigmas.weights(0).w_cov, w0);
    EXPECT_EQ(sigmas.weights(1).w_mean, w0);
    EXPECT_EQ(sigmas.weights(1).w_cov, w0);
    EXPECT_NE(sigmas.weights(2).w_mean, w0);
    EXPECT_NE(sigmas.weights(2).w_cov, w0);
    EXPECT_NE(sigmas.weights(3).w_mean, w0);
    EXPECT_NE(sigmas.weights(3).w_cov, w0);

    sigmas.point(2, p, w1_mean, w1_cov);
    EXPECT_NE(sigmas.weights(0).w_mean, w1_mean);
    EXPECT_NE(sigmas.weights(0).w_cov, w1_cov);
    EXPECT_NE(sigmas.weights(1).w_mean, w1_mean);
    EXPECT_NE(sigmas.weights(1).w_cov, w1_cov);
    EXPECT_EQ(sigmas.weights(2).w_mean, w1_mean);
    EXPECT_EQ(sigmas.weights(2).w_cov, w1_cov);
    EXPECT_NE(sigmas.weights(3).w_mean, w1_mean);
    EXPECT_NE(sigmas.weights(3).w_cov, w1_cov);

    sigmas.point(3,
                 p,
                 typename fl::Traits<Distribution>::Weight{w1_mean, w1_cov});
    EXPECT_NE(sigmas.weights(0).w_mean, w1_mean);
    EXPECT_NE(sigmas.weights(0).w_cov, w1_cov);
    EXPECT_NE(sigmas.weights(1).w_mean, w1_mean);
    EXPECT_NE(sigmas.weights(1).w_cov, w1_cov);
    EXPECT_EQ(sigmas.weights(2).w_mean, w1_mean);
    EXPECT_EQ(sigmas.weights(2).w_cov, w1_cov);
    EXPECT_EQ(sigmas.weights(3).w_mean, w1_mean);
    EXPECT_EQ(sigmas.weights(3).w_cov, w1_cov);
}

template <typename Distribution>
void weight_setter_tests(Distribution& sigmas)
{
    typename fl::Traits<Distribution>::Point p;

    const double w0 = 0.554;
    const double w1_mean = 0.6425434;
    const double w1_cov = 0.7325;

    sigmas.weight(1, w0);
    EXPECT_NE(sigmas.weights(0).w_mean, w0);
    EXPECT_NE(sigmas.weights(0).w_cov, w0);
    EXPECT_EQ(sigmas.weights(1).w_mean, w0);
    EXPECT_EQ(sigmas.weights(1).w_cov, w0);
    EXPECT_NE(sigmas.weights(2).w_mean, w0);
    EXPECT_NE(sigmas.weights(2).w_cov, w0);
    EXPECT_NE(sigmas.weights(3).w_mean, w0);
    EXPECT_NE(sigmas.weights(3).w_cov, w0);

    sigmas.weight(2, w1_mean, w1_cov);
    EXPECT_NE(sigmas.weights(0).w_mean, w1_mean);
    EXPECT_NE(sigmas.weights(0).w_cov, w1_cov);
    EXPECT_NE(sigmas.weights(1).w_mean, w1_mean);
    EXPECT_NE(sigmas.weights(1).w_cov, w1_cov);
    EXPECT_EQ(sigmas.weights(2).w_mean, w1_mean);
    EXPECT_EQ(sigmas.weights(2).w_cov, w1_cov);
    EXPECT_NE(sigmas.weights(3).w_mean, w1_mean);
    EXPECT_NE(sigmas.weights(3).w_cov, w1_cov);

    sigmas.weight(3, typename fl::Traits<Distribution>::Weight{w1_mean,w1_cov});
    EXPECT_NE(sigmas.weights(0).w_mean, w1_mean);
    EXPECT_NE(sigmas.weights(0).w_cov, w1_cov);
    EXPECT_NE(sigmas.weights(1).w_mean, w1_mean);
    EXPECT_NE(sigmas.weights(1).w_cov, w1_cov);
    EXPECT_EQ(sigmas.weights(2).w_mean, w1_mean);
    EXPECT_EQ(sigmas.weights(2).w_cov, w1_cov);
    EXPECT_EQ(sigmas.weights(3).w_mean, w1_mean);
    EXPECT_EQ(sigmas.weights(3).w_cov, w1_cov);
}


TEST(PointSet, weight_setter_fixed)
{
    constexpr size_t dim = 5;
    typedef Eigen::Matrix<double, 5, 1> Point ;

    {
        SCOPED_TRACE("using point setter");
        fl::PointSet<Point> sigmas(dim, 2*Point::SizeAtCompileTime + 1);
        point_weight_setter_tests(sigmas);
    }

    {
        SCOPED_TRACE("using weight setter");
        fl::PointSet<Point> sigmas(dim, 2*Point::SizeAtCompileTime + 1);
        weight_setter_tests(sigmas);
    }
}

TEST(PointSet, weight_setter_dynamic_constructor)
{
    constexpr size_t dim = 5;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Point ;

    {
        SCOPED_TRACE("using point setter");
        fl::PointSet<Point> sigmas(dim, 2 * dim + 1);
        point_weight_setter_tests(sigmas);
    }

    {
        SCOPED_TRACE("using weight setter");
        fl::PointSet<Point> sigmas(dim, 2 * dim + 1);
        weight_setter_tests(sigmas);
    }
}

TEST(PointSet, weight_setter_dynamic_dimension_setter)
{
    const size_t dim = 5;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Point ;

    {
        SCOPED_TRACE("using point setter");
        fl::PointSet<Point> sigmas(dim, 2 * dim + 1);
        sigmas.dimension(5);
        point_weight_setter_tests(sigmas);
    }

    {
        SCOPED_TRACE("using weight setter");
        fl::PointSet<Point> sigmas(dim, 2 * dim + 1);
        sigmas.dimension(5);
        weight_setter_tests(sigmas);
    }
}

TEST(PointSet, uninitialized_dynamic_dimension)
{
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Point ;

    fl::PointSet<Point, 10> sigmas;
    EXPECT_THROW(sigmas.point(0), fl::ZeroDimensionException);
    EXPECT_THROW(sigmas.weight(0), fl::ZeroDimensionException);
    EXPECT_THROW(sigmas.weight(0), fl::ZeroDimensionException);
}


template <typename Distribution>
void centered_points_tests(Distribution& sigmas)
{
    typedef typename fl::Traits<Distribution>::Point Point;
    typedef typename fl::Traits<Distribution>::PointMatrix PointMatrix;

    EXPECT_TRUE(sigmas.count_points() > 0);
    EXPECT_TRUE(sigmas.dimension() > 0);

    for (size_t i = 0; i < sigmas.count_points(); ++i)
    {
        sigmas.point(i, Point::Random(sigmas.dimension()));
    }

    PointMatrix centered_points = sigmas.centered_points();

    EXPECT_EQ(centered_points.cols(), sigmas.count_points());

    Point mean = Point::Zero(sigmas.dimension());
    for (size_t i = 0; i < centered_points.cols(); ++i)
    {
        mean += centered_points.col(i);
    }

    EXPECT_TRUE(mean.isZero());
}

TEST(PointSet, centered_points_fixed_fixed)
{
    typedef Eigen::Matrix<double, 10, 1> Point;
    typedef fl::PointSet<Point, 21> Distribution;

    Distribution sigmas;
    centered_points_tests(sigmas);
}

TEST(PointSet, centered_points_fixed_dynamic)
{
    constexpr size_t dim = 10;

    typedef Eigen::Matrix<double, dim, 1> Point;
    typedef fl::PointSet<Point> Distribution;

    {
        SCOPED_TRACE("Constructor");
        Distribution sigmas(dim, 21);
        centered_points_tests(sigmas);
    }

    {
        SCOPED_TRACE("Setters");
        Distribution sigmas;
        sigmas.resize(21);
        centered_points_tests(sigmas);
    }
}

TEST(PointSet, centered_points_dynamic_fixed)
{
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Point;
    typedef fl::PointSet<Point, 21> Distribution;

    Distribution sigmas;
    sigmas.dimension(10);
    centered_points_tests(sigmas);
}

TEST(PointSet, centered_points_dynamic_dynamic)
{
    constexpr size_t dim = 10;

    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Point;
    typedef fl::PointSet<Point> Distribution;

    {
        SCOPED_TRACE("Constructor");
        Distribution sigmas(dim, 21);
        centered_points_tests(sigmas);
    }

    {
        SCOPED_TRACE("Setters");
        Distribution sigmas;
        sigmas.resize(21);
        sigmas.dimension(10);
        centered_points_tests(sigmas);
    }
}
