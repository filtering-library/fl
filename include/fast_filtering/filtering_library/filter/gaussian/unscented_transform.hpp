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
 * @author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * Max-Planck-Institute for Intelligent Systems,
 * University of Southern California
 */

#ifndef FL__FILTER__GAUSSIAN__UNSCENTED_TRANSFORM_HPP
#define FL__FILTER__GAUSSIAN__UNSCENTED_TRANSFORM_HPP

#include <fast_filtering/utils/traits.hpp>
#include <fast_filtering/filtering_library/filter/gaussian/point_set_transform.hpp>

namespace fl
{

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
    typedef typename Traits<PointSetGaussian_>::Point        Point;
    typedef typename Traits<PointSetGaussian_>::PointMatrix  PointMatrix;
    typedef typename Traits<PointSetGaussian_>::Weight       Weight;
    typedef typename Traits<PointSetGaussian_>::WeightVector WeightVector;
    typedef typename Traits<
                        typename Traits<PointSetGaussian_>::Base
                    >::Operator Operator;

public:
    /**
     * Creates a UnscentedTransform
     *
     * @param alpha     UT Scaling parameter alpha (distance to the mean)
     * @param beta      UT Scaling parameter beta  (2.0 is optimal for Gaussian)
     * @param kappa     UT Scaling parameter kappa (higher order parameter)
     */
    UnscentedTransform(double alpha, double beta, double kappa)
        : alpha_(alpha),
          beta_(beta),
          kappa_(kappa)
    { }

    /**
     * \copydoc PointSetTransform::forward(const Moments_&,
     *                                     PointSetGaussian_&) const
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
     */
    virtual void forward(const Gaussian_& gaussian,
                         size_t global_rank,
                         size_t rank_offset,
                         PointSetGaussian_& transform) const
    {
        Operator covarianceSqr = gaussian.SquareRoot();

        covarianceSqr *=
                std::sqrt(
                    (double(global_rank)
                     + alpha_ * alpha_ * (double(global_rank) + kappa_)
                     - double(global_rank)));

        Point point_shift;
        const Point& mean = gaussian.Mean();

        transform.point(0, mean); // TODO specify the weights!!!


        const size_t start_1 = 1;
        const size_t limit_1 = start_1 + rank_offset;
        const size_t start_2 = limit_1;
        const size_t limit_2 = start_2 + gaussian.Dimension();
        const size_t start_3 = limit_2;
        const size_t limit_3 = global_rank;

        for (size_t i = start_1; i < limit_1; ++i)
        {
            transform.point(i, mean);
            transform.point(global_rank + i, mean);
        }

        for (size_t i = start_2; i < limit_2; ++i)
        {
            point_shift = covarianceSqr.col(i - (rank_offset + 1));
            transform.point(i, mean + point_shift);
            transform.point(global_rank + i, mean - point_shift);
        }

        for (size_t i = start_3; i <= limit_3; ++i)
        {
            transform.point(i, mean);
            transform.point(global_rank + i, mean);
        }

        // single loop implementation.
//        for (size_t i = 1; i <= global_rank; ++i)
//        {
//            if (rank_offset + 1 <= i && i < rank_offset + 1 + covarianceSqr.rows())
//            {
//                point_shift = covarianceSqr.col(i - (rank_offset + 1));

//                transform.point(i, mean + point_shift);
//                transform.point(global_rank + i, mean - point_shift);
//            }
//            else
//            {
//                transform.point(i, mean);
//                transform.point(global_rank + i, mean);
//            }
//        }
    }

protected:
    double alpha_;
    double beta_;
    double kappa_;
};

}

#endif
