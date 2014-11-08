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
 * @author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * Max-Planck-Institute for Intelligent Systems, University of Southern California
 */

#include <gtest/gtest.h>

#include <memory.h>

#include <fast_filtering/filtering_library/exception/exception.hpp>
#include <fast_filtering/filtering_library/filter/gaussian/point_set_gaussian.hpp>


TEST(PointSetGaussian, fixed_fixed)
{
    fl::PointSetGaussian<Eigen::Matrix<double, 10, 1>, 20> sigmas;

    EXPECT_NO_THROW(sigmas.point(0));
    EXPECT_NO_THROW(sigmas.point(19));
    EXPECT_THROW(sigmas.point(20), fl::OutOfBoundsException);
    // sigmas.resize(5); // doesn't compile as expected
    EXPECT_EQ(sigmas.count_points(), 20);
}

template <typename Distribution>
void test_dynamic_default(Distribution& sigmas)
{
    EXPECT_EQ(sigmas.count_points(), 0);
    EXPECT_THROW(sigmas.point(0), fl::OutOfBoundsException);
    EXPECT_THROW(sigmas.point(19), fl::OutOfBoundsException);
    EXPECT_EQ(sigmas.count_points(), 0);
    sigmas.resize(5);
    EXPECT_NO_THROW(sigmas.point(0));
    EXPECT_NO_THROW(sigmas.point(4));
    EXPECT_THROW(sigmas.point(5), fl::OutOfBoundsException);
    EXPECT_EQ(sigmas.count_points(), 5);
}

template <typename Distribution>
void test_dynamic(Distribution& sigmas, size_t initial_size)
{
    EXPECT_THROW(sigmas.point(initial_size), fl::OutOfBoundsException);
    EXPECT_EQ(sigmas.count_points(), initial_size);
    sigmas.resize(5);
    EXPECT_NO_THROW(sigmas.point(0));
    EXPECT_NO_THROW(sigmas.point(4));
    EXPECT_THROW(sigmas.point(5), fl::OutOfBoundsException);
    EXPECT_EQ(sigmas.count_points(), 5);
}

TEST(PointSetGaussian, fixed_dynamic)
{
    {SCOPED_TRACE("fixed_dynamic_default");
        fl::PointSetGaussian<Eigen::Matrix<double, 10, 1>> sigmas;
        test_dynamic_default(sigmas);
    }

    {SCOPED_TRACE("fixed_dynamic_0");
        fl::PointSetGaussian<Eigen::Matrix<double, 10, 1>> sigmas(0);
        test_dynamic_default(sigmas);
    }

    {SCOPED_TRACE("fixed_dynamic");
        fl::PointSetGaussian<Eigen::Matrix<double, 10, 1>> sigmas(4);
        test_dynamic(sigmas, 4);
    }
}

TEST(PointSetGaussian, dynamic_dynamic)
{
    {SCOPED_TRACE("dynamic_dynamic_default");
        fl::PointSetGaussian<Eigen::Matrix<double, Eigen::Dynamic, 1>> sigmas;
        test_dynamic_default(sigmas);
    }

    {SCOPED_TRACE("dynamic_dynamic_0");
        fl::PointSetGaussian<Eigen::Matrix<double, Eigen::Dynamic, 1>> sigmas(0);
        test_dynamic_default(sigmas);
    }

    {SCOPED_TRACE("dynamic_dynamic");
        fl::PointSetGaussian<Eigen::Matrix<double, Eigen::Dynamic, 1>> sigmas(4);
        test_dynamic(sigmas, 4);
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

TEST(PointSetGaussian, point_setter_fixed)
{
    typedef Eigen::Matrix<double, 5, 1> Point ;

    fl::PointSetGaussian<Point> sigmas(2 * Point::SizeAtCompileTime + 1);

    Point p = Point::Random();
    point_setter_tests(sigmas, p, 5);
}

TEST(PointSetGaussian, point_setter_dynamic)
{
    const size_t dimension = 5;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Point ;

    fl::PointSetGaussian<Point> sigmas(2 * dimension + 1, dimension);

    Point p = Point::Random(dimension);
    point_setter_tests(sigmas, p, dimension);
}


template <typename Distribution>
void point_weight_setter_tests(Distribution& sigmas)
{
    typename fl::Traits<Distribution>::Point p;

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

    sigmas.point(
            3, p, typename fl::Traits<Distribution>::Weight{w1_mean, w1_cov});
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

    sigmas.weight(
            3, typename fl::Traits<Distribution>::Weight{w1_mean, w1_cov});
    EXPECT_NE(sigmas.weights(0).w_mean, w1_mean);
    EXPECT_NE(sigmas.weights(0).w_cov, w1_cov);
    EXPECT_NE(sigmas.weights(1).w_mean, w1_mean);
    EXPECT_NE(sigmas.weights(1).w_cov, w1_cov);
    EXPECT_EQ(sigmas.weights(2).w_mean, w1_mean);
    EXPECT_EQ(sigmas.weights(2).w_cov, w1_cov);
    EXPECT_EQ(sigmas.weights(3).w_mean, w1_mean);
    EXPECT_EQ(sigmas.weights(3).w_cov, w1_cov);
}


TEST(PointSetGaussian, weight_setter_fixed)
{
    typedef Eigen::Matrix<double, 5, 1> Point ;

    {
        SCOPED_TRACE("using point setter");
        fl::PointSetGaussian<Point> sigmas(2*Point::SizeAtCompileTime + 1);
        //sigmas.Dimension(5); // doesn't compile as expected
        point_weight_setter_tests(sigmas);
    }

    {
        SCOPED_TRACE("using weight setter");
        fl::PointSetGaussian<Point> sigmas(2*Point::SizeAtCompileTime + 1);
        weight_setter_tests(sigmas);
    }
}

TEST(PointSetGaussian, weight_setter_dynamic_constructor)
{
    const size_t dimension = 5;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Point ;

    {
        SCOPED_TRACE("using point setter");
        fl::PointSetGaussian<Point> sigmas(2 * dimension + 1, dimension);
        point_weight_setter_tests(sigmas);
    }

    {
        SCOPED_TRACE("using weight setter");
        fl::PointSetGaussian<Point> sigmas(2 * dimension + 1, dimension);
        weight_setter_tests(sigmas);
    }
}

TEST(PointSetGaussian, weight_setter_dynamic_dimension_setter)
{
    const size_t dimension = 5;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Point ;

    {
        SCOPED_TRACE("using point setter");
        fl::PointSetGaussian<Point> sigmas(2 * dimension + 1);
        sigmas.Dimension(5);
        point_weight_setter_tests(sigmas);
    }

    {
        SCOPED_TRACE("using weight setter");
        fl::PointSetGaussian<Point> sigmas(2 * dimension + 1);
        sigmas.Dimension(5);
        weight_setter_tests(sigmas);
    }
}
