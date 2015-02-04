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
 * \file unscented_transform.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__FILTER__GAUSSIAN__RANDOM_SHITTY_TRANSFORM_HPP
#define FL__FILTER__GAUSSIAN__RANDOM_SHITTY_TRANSFORM_HPP

#include <fl/util/traits.hpp>
#include <fl/distribution/gaussian.hpp>
#include <fl/filter/gaussian/point_set_transform.hpp>

namespace fl
{

/**
 * \ingroup sigma_point_kalman_filters
 * \ingroup unscented_kalman_filter
 *
 * This is the Unscented Transform used in the Unscented Kalman Filter
 * \cite wan2000unscented . It implememnts the PointSetTransform interface.
 *
 * \copydetails PointSetTransform
 */
class RandomTransform
        : public PointSetTransform<RandomTransform>
{
public:
    /**
     * Creates a RandomTransform
     *
     * \param eta       Factor determining the number of point by
     *                  \f$\eta dim(X)\f$ where \f$X\f$ is the Gaussian variate
     */
    RandomTransform(double eta = 1.0)
        : PointSetTransform<RandomTransform>(this),
          eta_(eta)
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
        typedef typename Traits<PointSet_>::Point  Point;
        typedef typename Traits<PointSet_>::Weight Weight;

        const double dim = double(global_dimension);
        const size_t point_count = number_of_points(dim);
        const double w = 1./point_count;

        for (int i = 0; i < point_count; ++i)
        {
            point_set.point(i, gaussian.sample(), w);
        }
    }

    /**
     * \return Number of points generated by this transform
     *
     * \param dimension Dimension of the Gaussian
     */
    static constexpr size_t number_of_points(int dimension)
    {
        return (dimension != Eigen::Dynamic) ? 140  : -1;
    }

protected:
    /** \cond INTERNAL */
    double eta_;
    /** \endcond */
};

}

#endif