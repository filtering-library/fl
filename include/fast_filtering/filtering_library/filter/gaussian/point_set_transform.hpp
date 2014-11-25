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
 * @date 11/5/2014
 * @author Jan Issac (jan.issac@gmail.com)
 * Max-Planck-Institute for Intelligent Systems,
 * University of Southern California
 */

#ifndef FL__FILTER__GAUSSIAN__POINT_SET_TRANSFORM_HPP
#define FL__FILTER__GAUSSIAN__POINT_SET_TRANSFORM_HPP

#include <cstddef>
#include <fast_filtering/utils/meta.hpp>
#include <fast_filtering/distributions/gaussian.hpp>
#include <fast_filtering/filtering_library/filter/gaussian/point_set_gaussian.hpp>

namespace fl
{

/**
 * \brief The PointSetTransform is the interface of any transform that
 * transforms a Gaussian or the first two moments of a distribution into a set
 * of points (a statistic) representing the moments exactly. A typical transform
 * is the well known Unscented Transform \cite wan2000unscented.
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
        PointSetGaussian<Point, Derived::number_of_points(1)> point_set;

        /**
         * - Asserts the existens of the forward function with the required
         * parameters \code
         *   void forward(const Gaussian& src, PointSetGaussian& dest) const
         * \endcode
         */
        //derived->forward(ff::Gaussian<Point>(), point_set);

        /**
         * - Asserts the existens of the forward function with the required
         * parameters \code
         *   void forward(const Gaussian& src,
         *                size_t global_dimension,
         *                size_t dimension_offset,
         *                PointSetGaussian& dest) const
         * \endcode
         */
        //derived->forward(ff::Gaussian<Point>(), 1, 0, point_set);

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
     * @param [out] transform   Destination PointSetGaussian containing the
     *                          selected set of points represeting the source
     *                          Gaussian.
     */
    template <typename Gaussian, typename PointSetGaussian>
    void forward(const Gaussian& gaussian,
                 PointSetGaussian& transform) const;

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
     * @param [out] transform        Destination PointSetGaussian containing the
     *                               selected set of points represeting the
     *                               source Gaussian.
     */
    template <typename Gaussian, typename PointSetGaussian>
    void forward(const Gaussian& gaussian,
                 size_t global_dimension,
                 size_t dimension_offset,
                 PointSetGaussian& transform) const;
#endif
};

}

#endif
