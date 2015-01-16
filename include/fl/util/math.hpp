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
 * \ingroup math
 */
inline double sigmoid(const double& x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

/**
 * \brief log odd
 * \ingroup math
 */
inline double logit(const double& x)
{
    return std::log(x / (1.0 - x));
}

/**
 * \brief Constructs the QuaternionMatrix for the specified quaternion vetcor
 * \ingroup math
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
 * \ingroup math
 *
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

/**
 * Inverse of the error function.
 * \ingroup math
 *
 * \tparam RealType Argument and result type
 *
 * \return evaluates the erfinv at \f$ x \in (-1; 1) \f$
 */
template <typename RealType> RealType erfinv(RealType x);

/**
 * Single precision implementation of erfinv according to
 * \cite giles2010approximating
 * \ingroup math
 *
 * \return evaluates the erfinv at \f$ x \in (-1; 1) \f$
 */
template <> float erfinv<float>(float x)
{
    float w, p;
    w = - std::log((1.0f-x)*(1.0f+x));
    if ( w < 5.000000f )
    {
        w = w - 2.500000f;
        p =  2.81022636e-08f;
        p =  3.43273939e-07f + p * w;
        p =  -3.5233877e-06f + p * w;
        p = -4.39150654e-06f + p * w;
        p =   0.00021858087f + p * w;
        p =  -0.00125372503f + p * w;
        p =  -0.00417768164f + p * w;
        p =     0.246640727f + p * w;
        p =      1.50140941f + p * w;
    }
    else
    {
        w = std::sqrt(w) - 3.000000f;
        p = -0.000200214257f;
        p =  0.000100950558f + p * w;
        p =   0.00134934322f + p * w;
        p =  -0.00367342844f + p * w;
        p =   0.00573950773f + p * w;
        p =   -0.0076224613f + p * w;
        p =   0.00943887047f + p * w;
        p =      1.00167406f + p * w;
        p =      2.83297682f + p * w;
    }

    return p*x;
}

/**
 * Single precision implementation of erfinv according to
 * \cite giles2010approximating
 *
 * \ingroup math
 *
 * \return evaluates the erfinv at \f$ x \in (-1; 1) \f$
 */
template <> double erfinv<double>(double x)
{
    double w, p;

    w = - std::log((1.0-x)*(1.0+x));

    if ( w < 6.250000 )
    {
        w = w - 3.125000;
        p =  -3.6444120640178196996e-21;
        p =   -1.685059138182016589e-19 + p * w;
        p =   1.2858480715256400167e-18 + p * w;
        p =    1.115787767802518096e-17 + p * w;
        p =   -1.333171662854620906e-16 + p * w;
        p =   2.0972767875968561637e-17 + p * w;
        p =   6.6376381343583238325e-15 + p * w;
        p =  -4.0545662729752068639e-14 + p * w;
        p =  -8.1519341976054721522e-14 + p * w;
        p =   2.6335093153082322977e-12 + p * w;
        p =  -1.2975133253453532498e-11 + p * w;
        p =  -5.4154120542946279317e-11 + p * w;
        p =    1.051212273321532285e-09 + p * w;
        p =  -4.1126339803469836976e-09 + p * w;
        p =  -2.9070369957882005086e-08 + p * w;
        p =   4.2347877827932403518e-07 + p * w;
        p =  -1.3654692000834678645e-06 + p * w;
        p =  -1.3882523362786468719e-05 + p * w;
        p =    0.0001867342080340571352 + p * w;
        p =  -0.00074070253416626697512 + p * w;
        p =   -0.0060336708714301490533 + p * w;
        p =      0.24015818242558961693 + p * w;
        p =       1.6536545626831027356 + p * w;
    }
    else if ( w < 16.000000 )
    {
        w = std::sqrt(w) - 3.250000;
        p =   2.2137376921775787049e-09;
        p =   9.0756561938885390979e-08 + p * w;
        p =  -2.7517406297064545428e-07 + p * w;
        p =   1.8239629214389227755e-08 + p * w;
        p =   1.5027403968909827627e-06 + p * w;
        p =   -4.013867526981545969e-06 + p * w;
        p =   2.9234449089955446044e-06 + p * w;
        p =   1.2475304481671778723e-05 + p * w;
        p =  -4.7318229009055733981e-05 + p * w;
        p =   6.8284851459573175448e-05 + p * w;
        p =   2.4031110387097893999e-05 + p * w;
        p =   -0.0003550375203628474796 + p * w;
        p =   0.00095328937973738049703 + p * w;
        p =   -0.0016882755560235047313 + p * w;
        p =    0.0024914420961078508066 + p * w;
        p =   -0.0037512085075692412107 + p * w;
        p =     0.005370914553590063617 + p * w;
        p =       1.0052589676941592334 + p * w;
        p =       3.0838856104922207635 + p * w;
    }
    else
    {
        w = std::sqrt(w) - 5.000000;
        p =  -2.7109920616438573243e-11;
        p =  -2.5556418169965252055e-10 + p * w;
        p =   1.5076572693500548083e-09 + p * w;
        p =  -3.7894654401267369937e-09 + p * w;
        p =   7.6157012080783393804e-09 + p * w;
        p =  -1.4960026627149240478e-08 + p * w;
        p =   2.9147953450901080826e-08 + p * w;
        p =  -6.7711997758452339498e-08 + p * w;
        p =   2.2900482228026654717e-07 + p * w;
        p =  -9.9298272942317002539e-07 + p * w;
        p =   4.5260625972231537039e-06 + p * w;
        p =  -1.9681778105531670567e-05 + p * w;
        p =   7.5995277030017761139e-05 + p * w;
        p =  -0.00021503011930044477347 + p * w;
        p =  -0.00013871931833623122026 + p * w;
        p =       1.0103004648645343977 + p * w;
        p =       4.8499064014085844221 + p * w;
    }

    return p*x;
}

}

#endif
