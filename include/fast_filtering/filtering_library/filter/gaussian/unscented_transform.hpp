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

#ifndef FL__FILTER__GAUSSIAN__UNSCENTED_TRANSFORM_HPP
#define FL__FILTER__GAUSSIAN__UNSCENTED_TRANSFORM_HPP

#include <fast_filtering/utils/traits.hpp>
#include <fast_filtering/distributions/gaussian.hpp>
#include <fast_filtering/filtering_library/filter/gaussian/point_set_transform.hpp>

namespace fl
{

// Forward declarations
template <typename PointSetGaussian_, typename Gaussian_>
class UnscentedTransform;

/**
 *
 */
template <typename PointSetGaussian_, typename Gaussian_>
struct Traits<UnscentedTransform<PointSetGaussian_, Gaussian_>>
{
    typedef typename Traits<PointSetGaussian_>::Point  Point;
    typedef typename Traits<PointSetGaussian_>::Weight Weight;
    typedef typename ff::Traits<
                        typename Traits<PointSetGaussian_>::Base
                     >::Operator Operator;

    enum
    {
        NumberOfPoints = IsFixed<Point::RowsAtCompileTime>()
                           ? 2 * Point::RowsAtCompileTime + 1
                           : Eigen::Dynamic
    };
};



/**
 * This is the Unscented Transform used in the Unscented Kalman Filter
 * \cite wan2000unscented. It implememnts the PointSetTransform interface.
 *
 * \copydetails PointSetTransform
 */
template <typename PointSetGaussian_, typename Gaussian_ = PointSetGaussian_>
class UnscentedTransform:
        public PointSetTransform<PointSetGaussian_, Gaussian_>
{
public:
    typedef UnscentedTransform<PointSetGaussian_, Gaussian_> This;

    typedef typename Traits<This>::Point Point;
    typedef typename Traits<This>::Weight Weight;
    typedef typename Traits<This>::Operator Operator;

public:
    /**
     * Creates a UnscentedTransform
     *
     * @param alpha     UT Scaling parameter alpha (distance to the mean)
     * @param beta      UT Scaling parameter beta  (2.0 is optimal for Gaussian)
     * @param kappa     UT Scaling parameter kappa (higher order parameter)
     */
    UnscentedTransform(double alpha = 1., double beta = 2., double kappa = 0.)
        : alpha_(alpha),
          beta_(beta),
          kappa_(kappa)
    {
        /**
         * \internal
         *
         * assert the implication:
         *  isFixed(TransformNumberOfPoints) AND isFixed(PointSetNumberOfPoints)
         *  => TransformNumberOfPoints EQUAL PointSetNumberOfPoints
         */
        enum
        {
            TransformNumberOfPoints = Traits<This>::NumberOfPoints,
            PointSetNumberOfPoints = Traits<PointSetGaussian_>::NumberOfPoints
        };
        static_assert(!(IsFixed<TransformNumberOfPoints>() &&
                        IsFixed<PointSetNumberOfPoints>()) ||
                      (TransformNumberOfPoints == PointSetNumberOfPoints),
                      "Incompatible number of points of the specified"
                      " fixed-size PointSetGaussian");
    }

    /**
     * \copydoc PointSetTransform::forward(const Moments_&,
     *                                     PointSetGaussian_&) const
     *
     * \throws WrongSizeException
     * \throws ResizingFixedSizeEntityException
     */
    virtual void forward(const Gaussian_& gaussian,
                         PointSetGaussian_& transform) const
    {
        forward(gaussian, gaussian.Dimension(), 0, transform);
    }

    /**
     * \copydoc PointSetTransform::forward(const Moments_&,
     *                                     size_t,
     *                                     size_t,
     *                                     PointSetGaussian_&) const
     *
     * \throws WrongSizeException
     * \throws ResizingFixedSizeEntityException
     */
    virtual void forward(const Gaussian_& gaussian,
                         size_t global_rank,
                         size_t rank_offset,
                         PointSetGaussian_& transform) const
    {
        const double dim = double(global_rank);

        /**
         * \internal
         * A PointSetGaussian with a fixed number of points must have the
         * correct number of points which is required by this transform
         */
        if (IsFixed<Traits<PointSetGaussian_>::NumberOfPoints>() &&
            Traits<PointSetGaussian_>::NumberOfPoints != number_of_points(dim))
        {
            BOOST_THROW_EXCEPTION(
                WrongSizeException("Incompatible number of points of the"
                                   " specified fixed-size PointSetGaussian"));
        }

        transform.resize(number_of_points(dim));

        Operator covarianceSqr = gaussian.SquareRoot();
        covarianceSqr *= gamma_factor(dim);

        Point point_shift;
        const Point& mean = gaussian.Mean();

        const Weight weight_0{weight_mean_0(dim), weight_cov_0(dim)};
        const Weight weight_i{weight_mean_i(dim), weight_cov_i(dim)};

        transform.point(0, mean, weight_0);

        // use squential loops to enable loop unrolling
        const size_t start_1 = 1;
        const size_t limit_1 = start_1 + rank_offset;
        const size_t start_2 = limit_1;
        const size_t limit_2 = start_2 + gaussian.Dimension();
        const size_t start_3 = limit_2;
        const size_t limit_3 = global_rank;

        for (size_t i = start_1; i < limit_1; ++i)
        {
            transform.point(i, mean, weight_i);
            transform.point(global_rank + i, mean, weight_i);
        }

        for (size_t i = start_2; i < limit_2; ++i)
        {
            point_shift = covarianceSqr.col(i - (rank_offset + 1));

            transform.point(i, mean + point_shift, weight_i);
            transform.point(global_rank + i, mean - point_shift, weight_i);
        }

        for (size_t i = start_3; i <= limit_3; ++i)
        {
            transform.point(i, mean, weight_i);
            transform.point(global_rank + i, mean, weight_i);
        }
    }

    /**
     * @return Number of points generated by this transform
     *
     * @param global_rank Dimension of the Gaussian
     */
    static constexpr size_t number_of_points(size_t global_rank)
    {
        return 2 * global_rank + 1;
    }

public:
    /** \cond INTERNAL */

    /**
     * @return First mean weight
     *
     * @param dim Dimension of the Gaussian
     */
    double weight_mean_0(double dim) const
    {

        return lambda_scalar(dim) / (dim + lambda_scalar(dim));
    }

    /**
     * @return First covariance weight
     *
     * @param dim Dimension of the Gaussian
     */
    double weight_cov_0(double dim) const
    {
        return weight_mean_0(dim) + (1 - alpha_ * alpha_ + beta_);
    }

    /**
     * @return i-th mean weight
     *
     * @param dimension Dimension of the Gaussian
     */
    double weight_mean_i(double dim) const
    {
        return 1. / (2. * (dim + lambda_scalar(dim)));
    }

    /**
     * @return i-th covariance weight
     *
     * @param dimension Dimension of the Gaussian
     */
    double weight_cov_i(double dim) const
    {
        return weight_mean_i(dim);
    }

    /**
     * @param dim Dimension of the Gaussian
     */
    double lambda_scalar(double dim) const
    {
        return alpha_ * alpha_ * (dim + kappa_) - dim;
    }

    /**
     * @param dim  Dimension of the Gaussian
     */
    double gamma_factor(double dim) const
    {
        return std::sqrt(dim + lambda_scalar(dim));
    }
    /** \endcond */

protected:
    /** \cond INTERNAL */
    double alpha_;
    double beta_;
    double kappa_;
    /** \endcond */
};

}

#endif
