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
 * \file point_set_transform.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__FILTER__GAUSSIAN__POINT_SET_TRANSFORM_HPP
#define FL__FILTER__GAUSSIAN__POINT_SET_TRANSFORM_HPP

#include <cstddef>
#include <fl/util/meta.hpp>
#include <fl/distribution/gaussian.hpp>
#include <fl/filter/gaussian/point_set.hpp>

namespace fl
{

/**
 * \ingroup sigma_point_kalman_filters
 *
 * \brief The PointSetTransform is the interface of any transform that
 * transforms a Gaussian or the first two moments of a distribution into a set
 * of points (a statistic) representing the moments exactly. A typical transform
 * is the well known Unscented Transform \cite wan2000unscented .
 *
 *
 * \tparam Derived  PointSetTransform implementation type which must provide the
 *                  interface stated below.
 */
template <typename Derived>
class PointSetTransform
{
public:
    /**
     * Asserts correct interface
     *
     * @param derived instance pointer
     */
    explicit PointSetTransform(const Derived* const derived)
    {
        typedef Eigen::Matrix<double, 1, 1> Point;

        /** \cond INTERNAL */
        /** \remarks */

        /**
         * - Assert the existence of the function
         * \code constexpr size_t number_of_points(int dimension) const \endcode
         */
        PointSet<Point, Derived::number_of_points(1)> point_set;

        /**
         * - Asserts the existens of the forward function with the required
         * parameters \code
         *   void forward(const Gaussian& src, PointSet& dest) const
         * \endcode
         */
        derived->forward(Gaussian<Point>(), point_set);

        /**
         * - Asserts the existens of the forward function with the required
         * parameters \code
         *   void forward(const Gaussian& src,
         *                size_t global_dimension,
         *                size_t dimension_offset,
         *                PointSet& dest) const
         * \endcode
         */
        derived->forward(Gaussian<Point>(), 1, 0, point_set);

        /** \endcond */
    }

#ifdef PARSED_BY_DOXYGEN
    /**
     * \note This function is not explicitly definded nor implemented within
     *       PointSetTransform. However, it is implicitly required to be
     *       implemented in the Derived class.
     *
     * Transforms the given Gaussian into a statistic represented as a set of
     * points.
     *
     * @param [in]  gaussian    Source Gaussian distribution
     * @param [out] point_set   Destination PointSet containing the
     *                          selected set of points represeting the source
     *                          Gaussian.
     */
    template <typename Gaussian, typename PointSet>
    void forward(const Gaussian& gaussian,
                 PointSet& point_set) const;

    /**
     * \note This function is not explicitly definded nor implemented within
     *       PointSetTransform. However, it is implicitly required to be
     *       implemented in the Derived class.
     *
     * Performs a local transform on a partition of a Gaussian.
     * The given marginal Gaussian is transformed into a statistic represented
     * as a set of points \f${\cal X}_i\f$. The given Gaussian is assumed to be
     * a marginal Gaussian of a larger joint Gaussian with a block diagonal
     * covariance matrix
     *
     * \f$
     *   \Sigma = \begin{bmatrix}
     *               \Sigma_1 & 0        & 0 \\
     *               0        & \ddots   & 0 \\
     *               0        & 0        & \Sigma_n
     *            \end{bmatrix}
     * \f$
     * where \c gaussian = \f$\Sigma_i\f$.
     *
     * The number points in \f${\cal X}\f$ is a function of\f$dim(\Sigma_i)\f$.
     * Let the full transform of \f$\Sigma\f$ be
     *
     * \f$
     *  {\cal X} = \begin{bmatrix}
     *                  {\cal X}_1 \\
     *                  \vdots \\
     *                  {\cal X}_n
     *              \end{bmatrix}
     * \f$
     *
     * This transform determines the subspace \f${\cal X}_i\f$ for the marginal
     * \f$\Sigma_i\f$.
     *
     * \note Note that performing a regular transform on
     * \f$\Sigma_i\f$ as the full covariance will result in a different and
     * a smaller set of points \f${\cal X}'_i\f$ with different \em weights.
     *
     * @param [in]  gaussian         Source Gaussian distribution
     * @param [in]  global_dimension Dimension of the global covariance
     * @param [in]  dimension_offset Dimension offset determining the position
     *                               of the current marginal Gaussian.
     * @param [out] point_set        Destination PointSet containing the
     *                               selected set of points represeting the
     *                               source Gaussian.
     */
    template <typename Gaussian, typename PointSet>
    void forward(const Gaussian& gaussian,
                 size_t global_dimension,
                 size_t dimension_offset,
                 PointSet& point_set) const;
#endif
};

}

#endif
