/*
 * This is part of the FL library, a C++ Bayesian filtering library
 * (https://github.com/filtering-library)
 *
 * Copyright (c) 2014 Jan Issac (jan.issac@gmail.com)
 * Copyright (c) 2014 Manuel Wuthrich (manuel.wuthrich@gmail.com)
 *
 * This Source Code Form is subject to the terms of the MIT License (MIT).
 * A copy of the license can be found in the LICENSE file distributed with this
 * source code.
 */

/**
 * \file unscented_transform.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__FILTER__GAUSSIAN__UNSCENTED_TRANSFORM_HPP
#define FL__FILTER__GAUSSIAN__UNSCENTED_TRANSFORM_HPP

#include <fl/util/traits.hpp>
#include <fl/util/descriptor.hpp>
#include <fl/distribution/gaussian.hpp>
#include <fl/filter/gaussian/transform/point_set_transform.hpp>
#include <fl/distribution/joint_distribution.hpp>

#include <functional>


namespace fl
{

/**
 * \ingroup point_set_transform
 *
 * This is the Unscented Transform used in the Unscented Kalman Filter
 * \cite wan2000unscented . It implememnts the PointSetTransform interface.
 */
class UnscentedTransform
    : public PointSetTransform<UnscentedTransform>,
      public Descriptor
{
public:
    /**
     * Creates a UnscentedTransform
     *
     * \param alpha     UT Scaling parameter alpha (distance to the mean)
     * \param beta      UT Scaling parameter beta  (2.0 is optimal for Gaussian)
     * \param kappa     UT Scaling parameter kappa (higher order parameter)
     */
    UnscentedTransform(Real alpha = 1.0, Real beta = 2., Real kappa = 0.)
        : PointSetTransform<UnscentedTransform>(this),
          alpha_(alpha),
          beta_(beta),
          kappa_(kappa)
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
     *                                     PointSet&) const
     *
     * \throws WrongSizeException
     * \throws ResizingFixedSizeEntityException
     */
    template <typename Gaussian_, typename PointSet_>
    void operator()(const Gaussian_& gaussian,
                    PointSet_& point_set) const
    {
        forward(gaussian, gaussian.dimension(), 0, point_set);
    }

    /**
     * \copydoc PointSetTransform::forward(const Gaussian&,
     *                                     int global_dimension,
     *                                     int dimension_offset,
     *                                     PointSet&) const
     *
     * \throws WrongSizeException
     * \throws ResizingFixedSizeEntityException
     */
    template <typename Gaussian_, typename PointSet_>
    void forward(const Gaussian_& gaussian,
                 int global_dimension,
                 int dimension_offset,
                 PointSet_& point_set) const
    {
        typedef typename Traits<PointSet_>::Point  Point;
        typedef typename Traits<PointSet_>::Weight Weight;

        const Real dim = Real(global_dimension);
        const int point_count = number_of_points(dim);

        assert(point_count > 0);

        /**
         * \internal
         *
         * \remark
         * A PointSet with a fixed number of points must have the
         * correct number of points which is required by this transform
         */
        if (IsFixed<Traits<PointSet_>::NumberOfPoints>() &&
            Traits<PointSet_>::NumberOfPoints != point_count)
        {
            fl_throw(
                WrongSizeException("Incompatible number of points of the"
                                   " specified fixed-size PointSet"));
        }

        // will resize of transform size is different from point count.
        point_set.resize(point_count);

        auto&& covariance_sqrt = gaussian.square_root() * gamma_factor(dim);

        Point point_shift;
        const Point& mean = gaussian.mean();

        // set the first point
        point_set.point(0, mean, Weight{weight_mean_0(dim), weight_cov_0(dim)});

        // compute the remaining points
        Weight weight_i{weight_mean_i(dim), weight_cov_i(dim)};

        // use squential loops to enable loop unrolling
        const int start_1 = 1;
        const int limit_1 = start_1 + dimension_offset;
        const int limit_2 = limit_1 + gaussian.dimension();
        const int limit_3 = global_dimension;

        for (int i = start_1; i < limit_1; ++i)
        {
            point_set.point(i, mean, weight_i);
            point_set.point(global_dimension + i, mean, weight_i);
        }

        for (int i = limit_1; i < limit_2; ++i)
        {
            point_shift = covariance_sqrt.col(i - dimension_offset - 1);
            point_set.point(i, mean + point_shift, weight_i);
            point_set.point(global_dimension + i, mean - point_shift, weight_i);
        }

        for (int i = limit_2; i <= limit_3; ++i)
        {
            point_set.point(i, mean, weight_i);
            point_set.point(global_dimension + i, mean, weight_i);
        }
    }

    /**
     * \copydoc PointSetTransform::forward(const Gaussian&,
     *                                     int global_dimension,
     *                                     int dimension_offset,
     *                                     PointSet&) const
     *
     * \throws WrongSizeException
     * \throws ResizingFixedSizeEntityException
     */
    template <typename Gaussian_, typename PointSet_>
    void operator()(const Gaussian_& gaussian,
                    int global_dimension,
                    int dimension_offset,
                    PointSet_& point_set) const
    {
        forward(gaussian, global_dimension, dimension_offset, point_set);
    }

    /**
     * \return Number of points generated by this transform
     *
     * \param dimension Dimension of the Gaussian
     */
    static constexpr int number_of_points(int dimension)
    {
        return (dimension != Eigen::Dynamic) ? 2 * dimension + 1 : Eigen::Dynamic;
    }

public:
    /** \cond INTERNAL */

    /**
     * \returninternalean weight
     *
     * \param dim Dimension of the Gaussian
     */
    Real weight_mean_0(Real dim) const
    {
        return lambda_scalar(dim) / (dim + lambda_scalar(dim));
    }

    /**
     * \return First covariance weight
     *
     * \param dim Dimension of the Gaussian
     */
    Real weight_cov_0(Real dim) const
    {
        return weight_mean_0(dim) + (1 - alpha_ * alpha_ + beta_);
    }

    /**
     * \return i-th mean weight
     *
     * \param dimension Dimension of the Gaussian
     */
    Real weight_mean_i(Real dim) const
    {
        return 1. / (2. * (dim + lambda_scalar(dim)));
    }

    /**
     * \return i-th covariance weight
     *
     * \param dimension Dimension of the Gaussian
     */
    Real weight_cov_i(Real dim) const
    {
        return weight_mean_i(dim);
    }

    /**
     * \param dim Dimension of the Gaussian
     */
    Real lambda_scalar(Real dim) const
    {
        return alpha_ * alpha_ * (dim + kappa_) - dim;
    }

    /**
     * \param dim  Dimension of the Gaussian
     */
    Real gamma_factor(Real dim) const
    {
        return std::sqrt(dim + lambda_scalar(dim));
    }
    /** \endcond */

    virtual std::string name() const
    {
        return "UnscentedTransform";
    }

    virtual std::string description() const
    {
        return name();
    }

protected:
    /** \cond INTERNAL */
    Real alpha_;
    Real beta_;
    Real kappa_;
    /** \endcond */
};

}

#endif
