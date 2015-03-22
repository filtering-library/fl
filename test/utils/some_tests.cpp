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
 * \file some_tests.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

// this file should not remain in here

#include <gtest/gtest.h>

#define EIGEN_RUNTIME_NO_MALLOC
#include <Eigen/Dense>

#include <fl/util/math.hpp>
#include <fl/util/traits.hpp>

#include <cmath>
#include <utility>
// #include <boost/math/special_functions/erf.hpp>

static constexpr int dim = 500;
static constexpr int iterations = 2;

typedef Eigen::Matrix<double, dim, dim, Eigen::AutoAlign> Mat;
typedef Eigen::Matrix<double, -1, -1, Eigen::AutoAlign, dim, dim> DMat;

template <typename M>
struct C
{
    C() = default;

    C(const M& m) : m_(m) { }
    M m_;
};

template <typename M>
struct D
{
    D(M&& m) { m_.swap(m); }


    D(D const&) = delete;
    D& operator =(D const&) = delete;

    D() = default;
    D(D&& d) { m_.swap(d.m_); }
    D& operator =(D&& d)
    {
        m_.swap(d.m_);

        return *this;
    }

    M m_;
};

Mat R = Mat::Random();
Mat r = Mat::Zero();

TEST(MoveStuff, copy_stack)
{
    Eigen::internal::set_is_malloc_allowed(false);

    r = Mat::Zero();
    C<Mat> c;
    for (int i = 0; i < iterations; ++i)
    {
        Mat m = R;
        c = C<Mat>(m);
        r += (c.m_ * c.m_.transpose()).eval();
    }
}

TEST(MoveStuff, move_stack)
{
    Eigen::internal::set_is_malloc_allowed(false);

    r = Mat::Zero();
    D<Mat> d;
    for (int i = 0; i < iterations; ++i)
    {
        Mat m = R;
        d = D<Mat>(std::move(m));
        r += (d.m_ * d.m_.transpose()).eval();
    }
}

DMat DR = DMat::Random(dim, dim);
DMat dr = DMat::Zero(dim, dim);

TEST(MoveStuff, copy_heap)
{
    Eigen::internal::set_is_malloc_allowed(false);

    dr = DMat::Zero(dim, dim);
    C<DMat> c;
    for (int i = 0; i < iterations; ++i)
    {
        DMat m = DR;
        c = C<DMat>(m);
        dr += (c.m_ * c.m_.transpose()).eval();
    }
}

TEST(MoveStuff, move_heap)
{
    Eigen::internal::set_is_malloc_allowed(false);

    dr = DMat::Zero(dim, dim);
    D<DMat> d;

    for (int i = 0; i < iterations; ++i)
    {
        DMat m = DR;
        d = D<DMat>(std::move(m));
        dr += (d.m_ * d.m_.transpose()).eval();
    }
}








//static constexpr size_t number_of_points(int dimension)
//{
//    return (dimension > Eigen::Dynamic) ? 2 * dimension + 1 : 0;
//}

//TEST(conststuff, constexpressions)
//{
//    enum
//    {
//        JointDimension = fl::JoinSizes<1, 2, 3, 4>::Size,
//        NumberOfPoints = number_of_points(fl::JoinSizes<1, 2, 3, 4>::Size)
//    };

//    std::cout << JointDimension << std::endl;
//    std::cout << NumberOfPoints << std::endl;
//}

//float step_size_f =  0.000001f;
//double step_size_d = 0.000001;

//TEST(ErfInverse, comp_sp)
//{
//    long int iterations = 0;

//    for (float i = -0.9f; i < 0.9f; i+= step_size_f)
//    {
//        EXPECT_FLOAT_EQ(boost::math::erf_inv(i), fl::erfinv(i));

//        iterations++;
//    }

//    std::cout << "iterations  " << iterations << std::endl;
//}

//TEST(ErfInverse, boost_sp)
//{
//    float r;

//    for (float i = -0.9f; i < 0.9f; i+= step_size_f)
//    {
//        r = boost::math::erf_inv(i);
//    }
//}

//TEST(ErfInverse, giles_sp)
//{
//    float r;

//    for (float i = -0.9f; i < 0.9f; i+= step_size_f)
//    {
//        r = fl::erfinv(i);
//    }
//}

//TEST(ErfInverse, comp_dp)
//{
//    for (double i = -0.9; i < 0.9; i+= step_size_d)
//    {
//        EXPECT_FLOAT_EQ(boost::math::erf_inv(i), fl::erfinv(i));
//    }
//}

//TEST(ErfInverse, boost_dp)
//{
//    double r;

//    for (double i = -0.9; i < 0.9; i+= step_size_d)
//    {
//        r = boost::math::erf_inv(i);
//    }
//}

//TEST(ErfInverse, giles_dp)
//{
//    double r;

//    for (double i = -0.9; i < 0.9; i+= step_size_d)
//    {
//        r = fl::erfinv(i);
//    }
//}


//template <typename T> static T upper_incomplete_gamma(const T& s, const T& x) {
//    static const T T_0 = T(0), T_1 = T(1), T_2 = T(2), T_3 = T(3);

//    T A_prev = T_0;
//    T B_prev = T_1;
//    T A_cur  = pow(x, s) / exp(x);
//    T B_cur  = x - s + T_1;

//    T a = s - T_1;
//    T b = B_cur + T_2;
//    T n = s - T_3;

//    for(;;) {
//        const T A_next = b * A_cur + a * A_prev;
//        const T B_next = b * B_cur + a * B_prev;

//        if(A_next * B_cur == A_cur * B_next) {
//            return A_cur / B_cur;
//        }

//        A_prev = A_cur;
//        A_cur = A_next;

//        B_prev = B_cur;
//        B_cur = B_next;

//        a = a + n;
//        b = b + T_2;
//        n = n - T_2;
//    }
//}



//double upper_incomplete_gammaX(double s, const double t) {
//    double A_prev = 0;
//    double B_prev = 1;
//    double A_cur  = pow(t, s) / exp(t);
//    double B_cur  = t - s + 1;

//    double a = s - 1;
//    double b = B_cur + 2;
//    double n = s - 3;

//    while(true)
//    {
//        double A_next = b * A_cur + a * A_prev;
//        double B_next = b * B_cur + a * B_prev;

//        if(A_next * B_cur == A_cur * B_next) return A_cur / B_cur;

//        A_prev = A_cur;
//        A_cur = A_next;

//        B_prev = B_cur;
//        B_cur = B_next;

//        a = a + n;
//        b = b + 2;
//        n = n - 2;
//    }
//}


//TEST(TGamma, RT1)
//{
//    std::cout << boost::math::tgamma(0.00000000001, 2.0 * 0.033) << std::endl;
////    std::cout << upper_incomplete_gammaX(0.00000000001, 2.0  * 0.033)  << std::endl;
////    std::cout << fl::igamma(0.00000000001, 2.0  * 0.033)  << std::endl;
////    std::cout << fl::gser(0.00000000001, 2.0  * 0.033)  << std::endl;
//    std::cout << fl::exponential_integral(2.0 * 0.033)  << std::endl;
//    std::cout << fl::igamma(0., 2.0 * 0.033)  << std::endl;

////    std::cout << boost::math::tgamma(0.00000000001, 0.) << std::endl;
////    std::cout << upper_incomplete_gammaX(0.00000000001, 0.)  << std::endl;
////    std::cout << fl::igamma(0.00000000001, 0.)  << std::endl;
////    std::cout << fl::gser(0.00000000001, 0.)  << std::endl;
//}



//TEST(TGamma, speed_boost)
//{
//    double r = 0;

//    for (double i = 0.001; i < 10000; i+=0.001)
//    {
//        r += boost::math::tgamma(0.00000000001, i);
//    }

//    std::cout << r << std::endl;
//}


//TEST(TGamma, exponential_integral)
//{
//    double r = 0;

//    for (double i = 0.001; i < 10000; i+=0.001)
//    {
//        r += fl::exponential_integral(i);
//    }

//    std::cout << r << std::endl;
//}

//TEST(TGamma, igamma)
//{
//    double r = 0;

//    for (double i = 0.001; i < 10000; i+=0.001)
//    {
//        r += fl::igamma(0., i);
//    }

//    std::cout << r << std::endl;
//}


//TEST(TGamma, comp)
//{
//    for (double i = 0.001; i < 10000; i+=0.001)
//    {
//        EXPECT_NEAR(boost::math::tgamma(0.00000000001, i),
//                    fl::exponential_integral(i),
//                    1.e-9);

////        EXPECT_NEAR(boost::math::tgamma(0.00000000001, i),
////                    fl::igamma(0.0000000001, i),
////                    1.e-9);
////        std::cout << "i:" << i << " -> " <<
////                     boost::math::tgamma(0.0000000001, i) << " - " << fl::exponential_integral(i) <<
////                     " = " << boost::math::tgamma(0.0000000001, i) - fl::exponential_integral(i)<<std::endl;
////        std::cout << "i:" << i << " -> " <<
////                     boost::math::tgamma(0.0000000001, i) << " - " << fl::igamma(0.0000000001, i) <<
////                     " = " << boost::math::tgamma(0.0000000001, i) - fl::igamma(0.0000000001, i)<<std::endl<<std::endl;
//    }

//}

//TEST(TGamma, gcf)
//{
//    double r = 0;

//    for (int i = 0; i < 1000000; ++i)
//    {
//        r += fl::igamma(0.00000000001, 2.0 * 0.033+ i);
//    }

//    std::cout << r << std::endl;
//}

//TEST(TGamma, gcff)
//{
//    float r = 0;

//    for (int i = 0; i < 1000000; ++i)
//    {
//        r += fl::igamma(0.00000000001f, 2.0f * 0.033f + float(i));
//    }

//    std::cout << r << std::endl;
//}

//TEST(TGamma, speed_ranfcp)
//{
//    double r = 0;

//    for (int i = 0; i < 1000000; ++i)
//    {
//        r += upper_incomplete_gamma(0.00000000001, 2.0 *  0.033+ i);
//    }

//    std::cout << r << std::endl;
//}

//TEST(TGamma, speed_ranfcpX)
//{
//    double r = 0;

//    for (int i = 0; i < 1000000; ++i)
//    {
//        r += upper_incomplete_gammaX(0.00000000001, 2.0 * 0.033+ i);
//    }

//    std::cout << r << std::endl;
//}
