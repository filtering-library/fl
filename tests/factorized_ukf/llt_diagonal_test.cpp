

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

#include <Eigen/Dense>

#include <cmath>
#include <iostream>

size_t dia_dimension = 40;
size_t dia_iterations = 10000;
size_t dia_iterations_1x1 = 1e7;

TEST(Diags, diagAsDiagInv)
{
    Eigen::MatrixXd C = Eigen::MatrixXd::Identity(dia_dimension, dia_dimension) * 3;
    Eigen::MatrixXd R(dia_dimension, dia_dimension);

    for (size_t i = 0; i < 1000000; ++i)
    {
        R = C.diagonal().asDiagonal().inverse();
    }

    std::cout << R(0,0) << std::endl;
}


TEST(Diags, diagLoopInv)
{
    Eigen::MatrixXd C = Eigen::MatrixXd::Identity(dia_dimension, dia_dimension) * 3;
    Eigen::MatrixXd R(dia_dimension, dia_dimension);

    for (size_t i = 0; i < 1000000; ++i)
    {
        for (int j = 0; j < dia_dimension; ++j)
        {
            R(j , j) = 1./C(j, j);
        }
    }

    std::cout << R(0,0) << std::endl;
}


TEST(Diags, diaDiaLLT) /* remains in O(n^3) even though the matrix is diagonal */
{
    Eigen::MatrixXd C = (Eigen::MatrixXd::Ones(dia_dimension, 1)*25.).asDiagonal();

    Eigen::MatrixXd R;

    for (int i = 0; i < dia_iterations; i++)
    {
        R = C.llt().matrixL();
    }
}

TEST(Diags, diaElementWiseSqrt)  /* in O(n) */
{
    Eigen::MatrixXd C = Eigen::MatrixXd::Ones(dia_dimension, 1)*25.;
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(dia_dimension, dia_dimension);

    for (int i = 0; i < dia_iterations; i++)
    {
        for (int j = 0; j < dia_dimension; ++j)
        {
            R(j, j) = std::sqrt(C(j, 0));
        }
    }
}

TEST(Diags, diaDiaLLT1x1) /* remains in O(n^3) even though the matrix is diagonal */
{
    Eigen::MatrixXd C = Eigen::MatrixXd::Ones(1, 1)*25.;

    Eigen::MatrixXd R;

    for (int i = 0; i < dia_iterations_1x1; i++)
    {
        R = C.llt().matrixL();
    }
}

TEST(Diags, diaElementWiseSqrt1x1)  /* in O(n) */
{
    Eigen::MatrixXd C = Eigen::MatrixXd::Ones(1, 1)*25.;
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(1, 1);

    for (int i = 0; i < dia_iterations_1x1; i++)
    {
        for (int j = 0; j < 1; ++j)
        {
            R(j, j) = std::sqrt(C(j, 0));
        }
    }
}

TEST(Diags, isDiagonalSpeedTest) // too slow
{
    Eigen::MatrixXd C = Eigen::MatrixXd::Identity(6, 6);
    Eigen::MatrixXd R;

    for (int i = 0; i < dia_iterations_1x1; i++)
    {
        if (C.isDiagonal())
        {
             R = C;
        }
    }
}

double a_g = Eigen::Matrix<double, 1, 1>::Random()(0);
double b_g = Eigen::Matrix<double, 1, 1>::Random()(0);
double c_g = Eigen::Matrix<double, 1, 1>::Random()(0);
double d_g = Eigen::Matrix<double, 1, 1>::Random()(0);

double scalar_function(double a, double b, double c, double d)
{
    return (a * a * std::sqrt(b) - c) / std::pow(d, 3.42);
}

Eigen::Matrix<double, 1, 1>
vectorial_function(const Eigen::Matrix<double, 1, 1>& a,
                   const Eigen::Matrix<double, 1, 1>& b,
                   const Eigen::Matrix<double, 1, 1>& c,
                   const Eigen::Matrix<double, 1, 1>& d)
{
    return (a.transpose() * a * std::sqrt(b(0)) - c) / std::pow(d(0), 3.42);
}

TEST(OneDimensionalTests, scalar)
{
    double a, b, c, d;
    double r = 0;

    a = a_g;
    b = b_g;
    c = c_g;
    d = d_g;

    b *= b;

    for (size_t i = 0; i < 1e6; ++i)
    {
        r = scalar_function(a, b, c, d) + r;
    }
}

TEST(OneDimensionalTests, vectorial)
{
    Eigen::Matrix<double, 1, 1> a, b, c, d;
    Eigen::Matrix<double, 1, 1> r;

    a(0) = a_g;
    b(0) = b_g;
    c(0) = c_g;
    d(0) = d_g;
    r(0) = 0;

    b = b.transpose() * b;

    for (size_t i = 0; i < 1e6; ++i)
    {
        r = vectorial_function(a, b, c, d) + r;
    }
}

