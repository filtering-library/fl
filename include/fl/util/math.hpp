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
 * \file math.hpp
 * \date January 2015
 * \author Jan Issac (jan.issac@gmail.com)
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#ifndef FL__UTIL__MATH_HPP
#define FL__UTIL__MATH_HPP

#include <Eigen/Dense>

#include <cmath>
#include <vector>

namespace fl
{

/**
 * \brief Sigmoid function
 */
inline double sigmoid(const double& x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

/**
 * \brief log odd
 */
inline double logit(const double& x)
{
    return std::log(x / (1.0 - x));
}

/**
 * \brief Constructs the QuaternionMatrix for the specified quaternion vetcor
 *
 * \param q_xyzw  Quaternion vector
 *
 * \return Matrix representation of the quaternion vector
 */
inline Eigen::Matrix<double, 4, 3> QuaternionMatrix(
        const Eigen::Matrix<double, 4, 1>& q_xyzw)
{
    Eigen::Matrix<double, 4, 3> Q;
    Q <<	q_xyzw(3), q_xyzw(2), -q_xyzw(1),
            -q_xyzw(2), q_xyzw(3), q_xyzw(0),
            q_xyzw(1), -q_xyzw(0), q_xyzw(3),
            -q_xyzw(0), -q_xyzw(1), -q_xyzw(2);

    return 0.5*Q;
}

/**
 * Normalizes the values of input vector such that their sum is equal to the
 * specified \c sum. For instance, any convex combination requires that the
 * weights of the weighted sum sums up to 1.
 */
template <typename T> std::vector<T>
normalize(const std::vector<T>& input, T sum)
{
    T old_sum = 0;
    for(size_t i = 0; i < input.size(); i++)
        old_sum += input[i];
    T factor = sum/old_sum;

    std::vector<T> output(input.size());
    for(size_t i = 0; i < input.size(); i++)
        output[i] = factor*input[i];

    return output;
}

}

#endif
