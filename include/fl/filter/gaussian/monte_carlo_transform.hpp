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
 * \file monte_carlo_transform.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__FILTER__GAUSSIAN__MONTE_CARLO_TRANSFORM_HPP
#define FL__FILTER__GAUSSIAN__MONTE_CARLO_TRANSFORM_HPP

#include <fl/util/traits.hpp>
#include <fl/distribution/gaussian.hpp>
#include <fl/filter/gaussian/point_set_transform.hpp>

namespace fl
{

/**
 * \ingroup point_count_polices
 *
 * The \c MonomialPointCountPolicy determines the number of points as a function
 * of the dimension. The monimial function is of the form
 *
 * \f$ f(d) = a d^p + b\f$
 *
 * where \f$a\f$ is the parameter \c Factor, \f$p\f$ the parameter \c Power,
 * and \f$b\f$ the parameter \c Min.
 *
 * Strictly speaking the function \f$f(x)\f$ is a polynomial of the form
 * \f$g(d) = a d^p + b d^0\f$. However, for simplicity reasons, in most cases
 * the parameter \c Min is 0.
 */
template <size_t Power, size_t Factor = 1, size_t Min = 0>
class MonomialPointCountPolicy
{
public:
    static constexpr size_t number_of_points(int dimension)
    {
        return (dimension != Eigen::Dynamic)
                    ? monomial(dimension, Power, Factor, Min)
                    : Eigen::Dynamic;
    }

protected:
    static constexpr size_t monomial(size_t x, size_t p, size_t a, size_t b)
    {
        return (p > 0) ? a * x * monomial(x, p - 1, 1, 0) + b : 1;
    }
};

/**
 * \ingroup point_count_polices
 *
 * \c ConstantPointCountPolicy represents the policy for constant number of
 * points regardless the dimension
 *
 * \f$f(d) = b\f$
 *
 * where \f$b\f$ is the parameter \c NumberOfPoints.
 *
 * The policy is a special case of the
 * \c MonomialPointCountPolicy for the parameters
 *
 * \c MonomialPointCountPolicy<1, 0, \c NumberOfPoints>
 */
template <size_t NumberOfPoints>
struct ConstantPointCountPolicy
{
    static constexpr size_t number_of_points(int dimension)
    {
        return (dimension != Eigen::Dynamic)
                   ? MonomialPointCountPolicy<1, 0, NumberOfPoints>
                       ::number_of_points(dimension)
                   : Eigen::Dynamic;
    }
};

/**
 * \ingroup point_count_polices
 *
 * \c LinearPointCountPolicy represents the policy for a linear increase of
 * number of points w.r.t the dimension
 *
 * \f$ f(d) = a d + b\f$
 *
 * where \f$a\f$ is the parameter \c Factor and \f$b\f$ the parameter \c Min.
 *
 * The policy is a special case of the \c MonomialPointCountPolicy for the
 * parameters.
 *
 * \c MonomialPointCountPolicy<1, Factor, \c Min>
 */
template <size_t Factor = 1, size_t Min = 0>
struct LinearPointCountPolicy
{
    static constexpr size_t number_of_points(int dimension)
    {
        return (dimension != Eigen::Dynamic)
                   ? MonomialPointCountPolicy<1, Factor, Min>
                       ::number_of_points(dimension)
                   : Eigen::Dynamic;
    }
};


/**
 * \ingroup point_set_transform
 *
 * This is the Monte Carlo Transform (MCT) used in the Gaussian Filter, the
 * Sigma Point Kalman Filter, in the same sense as using the Unscented Transform
 * within the Unscented Kalman Filter. With a sufficient number of points, the
 * MCT is in practice able to capture relatively high order precision w.r.t the
 * Taylor series expansion. The
 * number of points is determined by the specified PointCountPolicy. The three
 * basic pre-implemented policies are \c MonomialPointCountPolicy,
 * \c ConstantPointCountPolicy, and \c LinearPointCountPolicy. The latter two
 * are a special case of the \c MonomialPointCountPolicy.
 */
template <
    typename PointCountPolicy = LinearPointCountPolicy<>
>
class MonteCarloTransform
        : public PointSetTransform<MonteCarloTransform<PointCountPolicy>>
{
public:
    /**
     * Creates a MonteCarloTransform
     */
    MonteCarloTransform()
        : PointSetTransform<MonteCarloTransform<PointCountPolicy>>(this)
    { }


    /**
     * \copydoc PointSetTransform::forward(const Gaussian&,
     *                                     PointSet&) const
     *
     * \throws WrongSizeException
     * \throws ResizingFixedSizeEntityException
     */
    template <typename Gaussian_, typename PointSet_>
    void forward(const Gaussian_& gaussian,
                 PointSet_& point_set) const
    {
        forward(gaussian, gaussian.dimension(), 0, point_set);
    }

    /**
     * \copydoc PointSetTransform::forward(const Gaussian&,
     *                                     size_t global_dimension,
     *                                     size_t dimension_offset,
     *                                     PointSet&) const
     *
     * \throws WrongSizeException
     * \throws ResizingFixedSizeEntityException
     */
    template <typename Gaussian_, typename PointSet_>
    void forward(const Gaussian_& gaussian,
                 size_t global_dimension,
                 size_t dimension_offset,
                 PointSet_& point_set) const
    {
//        typedef typename Traits<Gaussian_>::Variate NormalVariate;
//        typedef typename NormalVariate::Scalar Scalar;

//        static bool create_lookup_table = true;
//        static constexpr size_t table_size = 10000;
//        static StandardGaussian<NormalVariate> mvnd;
//        static Eigen::Matrix<
//                   Scalar,
//                   NormalVariate::RowsAtCompileTime,
//                   table_size
//               > lt;

//        if (create_lookup_table)
//        {
//            mvnd.dimension(gaussian.dimension());
//            lt.resize(gaussian.dimension(), table_size);

//            for (size_t i = 0; i < table_size; ++i)
//            {
//                lt.col(i) = mvnd.sample();
//            }

//            create_lookup_table = false;
//        }

//        auto&& cov_sqrt = gaussian.square_root();

        const size_t point_count = number_of_points(global_dimension);
        const double w = 1./double(point_count);        

        for (int i = 0; i < point_count; ++i)
        {
            point_set.point(i, gaussian.sample(), w);

            //point_set.point(i, cov_sqrt * lt.col(rand() % table_size), w);
        }
    }

    /**
     * \return Number of points generated by this transform
     *
     * \param dimension Dimension of the Gaussian
     */
    static constexpr size_t number_of_points(int dimension)
    {
        return PointCountPolicy::number_of_points(dimension);
    }
};

}

#endif
