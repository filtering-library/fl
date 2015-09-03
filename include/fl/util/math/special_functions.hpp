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
 */

#pragma once


#include <Eigen/Dense>

#include <cmath>
#include <vector>
#include <random>

#include <iostream>

#include <fl/util/types.hpp>

namespace fl
{

/**
 * Euler-Mascheroni Constant
 * \ingroup special_functions
 */
static constexpr double GAMMA =
        0.57721566490153286060651209008240243104215933593992;

/**
 * \brief Incomplete upper gamma function for positive \c a and \c z
* \ingroup special_functions
 *
 * This is the unnormalized incomplete upper gamma function implementing the
 * continued fraction representation \cite press2007numerical .
 *
 * \f$ \Gamma(a, z) = \int\limits^\infty_z t^{a-1}e^{-t}dt \f$
 *
 * \return \f$ \Gamma(a, z) \f$
 */
inline double igamma(const double a, const double z)
{
    static constexpr double EPS = std::numeric_limits<double>::epsilon();
    static constexpr double FPMIN = std::numeric_limits<double>::min() / EPS;

    if (z <= 0.) return std::numeric_limits<double>::quiet_NaN();

    int i;
    double an, b, c, d, del, h;

    b = z + 1.0 - a;
    c = 1.0 / FPMIN;
    d = 1.0 / b;
    h = d;

    for (i = 1; ; i++)
    {
        an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if (std::fabs(d) < FPMIN) d = FPMIN;
        c = b + an / c;
        if (std::fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        del = d * c;
        h *= del;

        if (std::fabs(del - 1.0) <= EPS) break;
    }

    return std::exp(-z + a * std::log(z)) * h;
}


//inline double inv_igamma_p(double a, double p)
//{
//    int j;
//    double x, err, t, u, pp, lna1, afac, a1 = a - 1;
//    const double EPS=1.e-8;

//    gln=gammln(a);

//    assert(a > 0.);

//    if (p >= 1.) return std::max(100., a + 100. * std::sqrt(a));
//    if (p <= 0.) return 0.0;
//    if (a > 1.)
//    {
//        lna1 = std::log(a1);
//        afac = std::exp(a1 * (lna1 - 1.) - gln);
//        pp = (p < 0.5)? p : 1. - p;
//        t = sqrt(-2.*log(pp));
//        x = (2.30753 + t * 0.27061) / (1. + t * (0.99229 + t * 0.04481)) - t;
//        if (p < 0.5) x = -x;
//        x = std::max(1.e-3, a * std::pow(1. - 1. / (9. * a) - x/(3.*std::sqrt(a)),3));
//    }
//    else
//    {
//        t = 1.0 - a*(0.253+a*0.12);

//        if (p < t)
//        {
//            x = std::pow(p/t,1./a);
//        }
//        else
//        {
//            x = 1.-std::log(1.-(p-t)/(1.-t));
//        }
//    }
//    for (j=0;j<12;j++)
//    {
//        if (x <= 0.0) return 0.0;

//        err = gammp(a,x) - p;
//        if (a > 1.)
//        {
//            t = afac*std::exp(-(x-a1)+a1*(std::log(x)-lna1));
//        }
//        else
//        {
//            t = std::exp(-x + a1 * std::log(x) - gln);
//        }
//        u = err/t;
//        x -= (t = u/(1. - 0.5 * std::min(1., u * ((a - 1.) / x - 1))));

//        if (x <= 0.)
//        {
//            x = 0.5*(x + t);
//        }

//        if (std::fabs(t) < EPS*x ) break;
//    }
//    return x;
//    }

//}


/**
 * \brief This is a deterministic exponential integral approximation
 *        \cite barry2000approximation .
* \ingroup special_functions
 *
 *
 * The exponential integral \f$E_1(z)\f$ is a special case of the upper
 * incomplete gamma function
 *
 * \f$ \Gamma(a, z) = \int\limits^\infty_z t^{a-1}e^{-t}dt \f$
 *
 * for \f$a = 0\f$ the gamma function is known as the exponential integral
 *
 * \f$ \Gamma(0, z) = \int\limits^\infty_z t^{-1}e^{-t}dt = E_1(z)\f$
 *
 * \return \f$E_1(z) = \Gamma(0, z)\f$
 * \note For high accuracy please use igamma(0, z)
 */
inline double exponential_integral(double z)
{
    if (z < 10.) return igamma(0., z);

    static constexpr double G = std::exp(-fl::GAMMA);
    static constexpr double b = std::sqrt((2. * (1.-G)) / (G * (2. - G)));
    static constexpr double h_inf = ((1.-G) * (std::pow(G, 2.) - 6. * G + 12.))
                                     / (3. * G * std::pow((2. - G), 2.) * b);

    static constexpr double q_frac = 20. / 47.;
    static constexpr double q_expx = std::sqrt(31. / 26.);

    const double q = q_frac * std::pow(z, q_expx);
    const double h = 1. / (1. + z * std::sqrt(z)) + (h_inf * q) / (1. + q);

    double E1 = std::exp(-z) *
                std::log(1. + G/z - (1.-G) / std::pow(h + b*z, 2.));
    E1 /= (G + (1. - G) * std::exp(-z / (1. - G)));

    return E1;
}

/**
 * Inverse of the error function.
* \ingroup special_functions
 *
 * \tparam RealType Argument and result type
 *
 * \return evaluates the erfinv at \f$ x \in (-1; 1) \f$
 */
template <typename RealType> inline RealType erfinv(RealType x);

/**
 * Single precision implementation of erfinv according to
 * \cite giles2010approximating
* \ingroup special_functions
 *
 * \return evaluates the erfinv at \f$ x \in (-1; 1) \f$
 */
template <> inline float erfinv<float>(float x)
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
 * \ingroup special_functions
 *
 * \return evaluates the erfinv at \f$ x \in (-1; 1) \f$
 */
template <> inline  double erfinv<double>(double x)
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


