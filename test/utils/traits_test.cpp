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
 * \file traits_test.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <fl/util/math.hpp>
#include <fl/util/traits.hpp>

#include <cmath>

TEST(TraitsTests, is_dynamic)
{
    EXPECT_TRUE(fl::IsDynamic<Eigen::Dynamic>());
    EXPECT_FALSE(fl::IsDynamic<10>());
}

TEST(TraitsTests, is_fixed)
{
    EXPECT_FALSE(fl::IsFixed<Eigen::Dynamic>());
    EXPECT_TRUE(fl::IsFixed<10>());
}

TEST(TraitsTests, DimensionOf_dynamic)
{
    EXPECT_EQ(fl::DimensionOf<Eigen::MatrixXd>(), 0);
    EXPECT_EQ(fl::DimensionOf<Eigen::VectorXd>(), 0);

    typedef Eigen::Matrix<double, 10, -1> PartiallyDynamicMatrix;
    EXPECT_EQ(fl::DimensionOf<PartiallyDynamicMatrix>(), 0);
}

TEST(TraitsTests, DimensionOf_fixed)
{
    EXPECT_EQ(fl::DimensionOf<Eigen::Matrix3d>(), 3);
    EXPECT_EQ(fl::DimensionOf<Eigen::Vector4d>(), 4);

    typedef Eigen::Matrix<double, 10, 10> FixedMatrix;
    EXPECT_EQ(fl::DimensionOf<FixedMatrix>(), 10);

    typedef Eigen::Matrix<double, 10, 1> FixedVector;
    EXPECT_EQ(fl::DimensionOf<FixedVector>(), 10);
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

template <typename Mat>
Mat identity()
{
    Mat m;
    m.setIdentity(100, 100);
    return m;
}

//template <typename Mat>
//void ref_identity(Mat& m)
//{
//    m.setIdentity();
//}

//template <typename Mat>
//const Mat const_identity()
//{
//    return Mat::Identity();
//}

TEST(Func, identity)
{
    typedef Eigen::Matrix<double, 100, 100> Mat;

    Mat r = Mat::Zero();

    for (int i = 0; i < 100000; ++i)
    {
        Mat m = identity<Mat>();
        r(0, 0) += m(0, 0);
    }

    std::cout << r(0, 0) << std::endl;
}


//TEST(Func, const_identity)
//{
//    typedef Eigen::Matrix<double, 100, 100> Mat;

//    Mat r = Mat::Zero();

//    for (int i = 0; i < 100000; ++i)
//    {
//        Mat m = const_identity<Mat>();
//        r(0, 0) += m(0, 0);
//    }

//    std::cout << r(0, 0) << std::endl;
//}


//TEST(Func, ref_identity)
//{
//    typedef Eigen::Matrix<double, 100, 100> Mat;

//    Mat r = Mat::Zero();

//    for (int i = 0; i < 100000; ++i)
//    {
//        Mat m;
//        ref_identity<Mat>(m);
//        r(0, 0) += m(0, 0);
//    }

//    std::cout << r(0, 0) << std::endl;
//}


TEST(Func, direct)
{
    typedef Eigen::Matrix<double, 100, 100> Mat;

    Mat r = Mat::Zero();

    for (int i = 0; i < 100000; ++i)
    {
        Mat m = Mat::Identity(100, 100);
        r(0, 0) += m(0, 0);
    }

    std::cout << r(0, 0) << std::endl;
}

//TEST(Func, direct2)
//{
//    typedef Eigen::Matrix<double, 100, 100> Mat;

//    Mat r = Mat::Zero();

//    for (int i = 0; i < 100000; ++i)
//    {
//        Mat m(100, 100);
//        m.setIdentity(100, 100);
//        r(0, 0) += m(0, 0);
//    }

//    std::cout << r(0, 0) << std::endl;
//}


//TEST(Func, direct2_dynamic)
//{
//    typedef Eigen::Matrix<double, -1, -1> Mat;

//    Mat r = Mat::Zero(100, 100);

//    for (int i = 0; i < 100000; ++i)
//    {
//        Mat m = Mat(100, 100);
//        m.setIdentity(100, 100);
//        r(0, 0) += m(0, 0);
//    }

//    std::cout << r(0, 0) << std::endl;
//}


//TEST(ResizeMats, resize)
//{
//    Eigen::Matrix<double, 10, 1> X;

//    std::cout << X.size() << std::endl;

//    X.setZero(20, 1);
//    X.resize(21, 1);
//    X.resize(20, 2);
//    X.resize(0, 2);

//    std::cout << X.size() << std::endl;
//}





