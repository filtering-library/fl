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
 * \file point_set.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__FILTER__GAUSSIAN__POINT_SET_HPP
#define FL__FILTER__GAUSSIAN__POINT_SET_HPP

#include <map>
#include <cmath>
#include <type_traits>

#include <Eigen/Dense>


#include <fl/util/meta.hpp>
#include <fl/util/traits.hpp>

#include <fl/distribution/gaussian.hpp>
#include <fl/exception/exception.hpp>
#include <fl/filter/filter_interface.hpp>

/** \cond INTERNAL */
/**
 * Checks dimensions
 */
#define INLINE_CHECK_POINT_SET_DIMENSIONS() \
    assert(points_.cols() == weights_.size()); \
    if (points_.rows() == 0) \
    { \
        BOOST_THROW_EXCEPTION(ZeroDimensionException("PointSet")); \
    } \
    if (points_.cols() == 0)  \
    { \
        BOOST_THROW_EXCEPTION(Exception("PointSet contains no points.")); \
    }

/**
 * Checks for index out of bounds and valid dimensions
 */
#define INLINE_CHECK_POINT_SET_BOUNDS(i) \
    if (i >= weights_.size()) \
    { \
        BOOST_THROW_EXCEPTION(OutOfBoundsException(i, weights_.size())); \
    } \
    INLINE_CHECK_POINT_SET_DIMENSIONS();
/** \endcond */

namespace fl
{

// Forward declaration
template <typename Point, int Points_> class PointSet;

/**
 * Trait struct of a PointSet
 */
template <typename Point_, int Points_>
struct Traits<PointSet<Point_, Points_>>
{    
    /**
     * \brief Point type
     */
    typedef Point_ Point;

    /**
     * \brief Number of points for fixed-size set
     */
    enum
    {
        /**
         * \brief Number of points which provided by the PointSet.
         *
         * If the number of points is unknown and there for dynamic, then
         * NumberOfPoints is set to Eigen::Dynamic
         */
        NumberOfPoints = IsFixed<Points_>() ? Points_ : Eigen::Dynamic
    };

    /**
     * \brief Weight harbors the weights of a point. For each of the first two
     * moments there is a separate weight.
     *
     *
     * Generally a single weight suffices. However, some transforms utilize
     * different weights for each moment to select a set of points representing
     * the underlying moments.
     */
    struct Weight
    {
        /**
         * First moment (mean) point weight
         */
        double w_mean;

        /**
         * Second centered moment (covariance) point weight
         */
        double w_cov;
    };

    /**
     * \brief Point container type
     *
     * \details
     * The point container type has a fixed-size dimension of a point
     * and the number of the points is statically known.
     */
    typedef Eigen::Matrix<
                typename Point::Scalar,
                Point::RowsAtCompileTime,
                Points_
            > PointMatrix;

    /**
     * \brief WeightVector
     */
    typedef Eigen::Matrix<
                typename Point::Scalar,
                Points_,
                1
            > WeightVector;

    /**
     * \brief Weight list of all points
     */
    typedef std::vector<Weight> Weights;
};

/**
 * \class PointSet
 *
 * PointSet represents a container of fixed-size or dynamic-size points each
 * paired with a set of weights. PointSet has two degree-of-freedoms. The first
 * is the dimension of the points. The second os the number of points within the
 * set. Each of the parameter can either be fixed at compile time or left
 * unspecified. That is, the parameter is set to Eigen::Dynamic.
 *
 * \tparam Point    Gaussian variable type
 * \tparam Points_  Number of points representing the gaussian
 */
template <typename Point_, int Points_ = Eigen::Dynamic>
class PointSet
{
public:
    typedef PointSet<Point_, Points_> This;

    typedef typename Traits<This>::Point        Point;
    typedef typename Traits<This>::PointMatrix  PointMatrix;
    typedef typename Traits<This>::Weight       Weight;
    typedef typename Traits<This>::Weights      Weights;
    typedef typename Traits<This>::WeightVector WeightVector;    

public:
    /**
     * Creates a PointSet
     *
     * \param points_count   Number of points representing the Gaussian
     * \param dimension      Sample space dimension
     */
    PointSet(size_t dimension = DimensionOf<Point>(),
             size_t points_count = MaxOf<Points_, 0>())
        : points_(dimension, points_count),
          weights_(points_count)
    {
        assert(points_count >= 0);
        static_assert(Points_ >= Eigen::Dynamic, "Invalid point count");

        points_.setZero();

        double weight = (points_count > 0) ? 1./double(points_count) : 0;
        std::fill(weights_.begin(), weights_.end(), Weight{weight, weight});
    }

    /**
     * \brief Overridable default constructor
     */
    virtual ~PointSet() { }

    /**
     * Resizes a dynamic-size PointSet to a desired. For
     *
     * \param Points_   Number of points
     *
     * \throws ResizingFixedSizeEntityException
     */
    void resize(size_t points_count)
    {
        if (weights_.size() == points_count) return;

        if (IsFixed<Points_>())
        {
            BOOST_THROW_EXCEPTION(
                fl::ResizingFixedSizeEntityException(weights_.size(),
                                                     points_count,
                                                     "poit set Gaussian"));
        }

        points_.setZero(points_.rows(), points_count);

        weights_.resize(points_count);
        const double weight = (points_count > 0)? 1. / double(points_count) : 0;
        std::fill(weights_.begin(), weights_.end(), Weight{weight, weight});
    }

    /**
     * Sets the new dimension for dynamic-size points (not to confuse
     * with the number of points)
     *
     * \param dim  Dimension of each point
     */
    void dimension(size_t dim)
    {
        points_.resize(dim, count_points());
    }

    /**
     * \return Dimension of containing points
     */
    size_t dimension() const
    {
        return points_.rows();
    }

    /**
     * \return The number of points
     */
    size_t count_points() const
    {
        return points_.cols();
    }

    /**
     * \return i-th point
     *
     * \param i Index of requested point
     *
     * \throws OutOfBoundsException
     * \throws ZeroDimensionException
     */
    Point point(size_t i) const
    {
        INLINE_CHECK_POINT_SET_BOUNDS(i);

        return points_.col(i);
    }

    /**
     * \return weight of i-th point assuming both weights are the same
     *
     * \param i Index of requested point
     *
     * \throws OutOfBoundsException
     * \throws ZeroDimensionException
     */
    double weight(size_t i)
    {
        INLINE_CHECK_POINT_SET_BOUNDS(i);

        return weights_[i].w_mean;
    }

    /**
     * \return weights of i-th point
     *
     * \param i Index of requested point
     *
     * \throws OutOfBoundsException
     * \throws ZeroDimensionException
     */
    const Weight& weights(size_t i) const
    {
        INLINE_CHECK_POINT_SET_BOUNDS(i);

        return weights_[i];
    }

    /**
     * \return Point matrix
     */
    const PointMatrix& points() const noexcept
    {
        return points_;
    }

    /**
     * \return point weights vector
     */
    const Weights& weights() const noexcept
    {
        return weights_;
    }

    /**
     * \return Returns the weights for the mean of the points as a vector
     */
    WeightVector mean_weights_vector() const noexcept
    {
        const size_t point_count = count_points();

        WeightVector weight_vec(point_count);

        for (size_t i = 0; i < point_count; ++i)
        {
            weight_vec(i) = weights_[i].w_mean;
        }

        return weight_vec;
    }

    /**
     * \return Returns the weights for the covariance of the points as a vector
     */
    WeightVector covariance_weights_vector() const noexcept
    {
        const size_t point_count = count_points();

        WeightVector weight_vec(point_count);

        for (size_t i = 0; i < point_count; ++i)
        {
            weight_vec(i) = weights_[i].w_cov;
        }

        return weight_vec;
    }

    /**
     * Sets a given point at position i
     *
     * \param i         Index of point
     * \param p         The new point
     *
     * \throws OutOfBoundsException
     * \throws ZeroDimensionException
     */
    void point(size_t i, Point p)
    {
        INLINE_CHECK_POINT_SET_BOUNDS(i);

        points_.col(i) = p;
    }

    /**
     * Sets a given point at position i along with its weights
     *
     * \param i         Index of point
     * \param p         The new point
     * \param w         Point weights. The weights determinaing the first two
     *                  moments are the same
     *
     * \throws OutOfBoundsException
     * \throws ZeroDimensionException
     */
    void point(size_t i, Point p, double w)
    {
        point(i, p, Weight{w, w});        
    }

    /**
     * Sets a given point at position i along with its weights
     *
     * \param i         Index of point
     * \param p         The new point
     * \param w_mean    point weight used to compute the first moment
     * \param w_cov     point weight used to compute the second centered moment
     *
     * \throws OutOfBoundsException
     * \throws ZeroDimensionException
     */
    void point(size_t i, Point p, double w_mean , double w_cov)
    {
        point(i, p, Weight{w_mean, w_cov});
    }

    /**
     * Sets a given point at given position i along with its weights
     *
     * \param i         Index of point
     * \param p         The new point
     * \param weights   point weights
     *
     * \throws OutOfBoundsException
     * \throws ZeroDimensionException
     */
    void point(size_t i, Point p, Weight weights)
    {
        INLINE_CHECK_POINT_SET_BOUNDS(i);

        points_.col(i) = p;
        weights_[i] = weights;
    }

    /**
     * Sets a given weight of a point at position i
     *
     * \param i         Index of point
     * \param w         Point weights. The weights determinaing the first two
     *                  moments are the same
     *
     * \throws OutOfBoundsException
     * \throws ZeroDimensionException
     */
    void weight(size_t i, double w)
    {
        weight(i, Weight{w, w});
    }

    /**
     * Sets given weights of a point at position i
     *
     * \param i         Index of point
     * \param w_mean    point weight used to compute the first moment
     * \param w_cov     point weight used to compute the second centered moment
     *
     * \throws OutOfBoundsException
     * \throws ZeroDimensionException
     */
    void weight(size_t i, double w_mean , double w_cov)
    {
        weight(i, Weight{w_mean, w_cov});
    }

    /**
     * Sets given weights of a point at position i
     *
     * \param i         Index of point
     * \param weights   point weights
     *
     * \throws OutOfBoundsException
     * \throws ZeroDimensionException
     */
    void weight(size_t i, Weight weights)
    {
        INLINE_CHECK_POINT_SET_BOUNDS(i);

        weights_[i] = weights;
    }

    /**
     * \return Centered points matrix.
     *
     * Creates a PointMatrix populated with zero mean points
     *
     * \throws ZeroDimensionException
     * \throws Exception
     */
    PointMatrix centered_points() const
    {
        INLINE_CHECK_POINT_SET_DIMENSIONS();

        PointMatrix centered(points_.rows(), points_.cols());

        const Point weighted_mean = mean();
        const size_t point_count = points_.cols();
        for (size_t i = 0; i < point_count; ++i)
        {
            centered.col(i) = points_.col(i) - weighted_mean;
        }

        return centered;
    }

    /**
     * \return The weighted mean of all points
     *
     * \throws ZeroDimensionException
     * \throws Exception
     */
    Point mean() const
    {        
        INLINE_CHECK_POINT_SET_DIMENSIONS();

        Point weighted_mean;
        weighted_mean.setZero(points_.rows());

        const size_t point_count = points_.cols();
        for (size_t i = 0; i < point_count; ++i)
        {
            weighted_mean += weights_[i].w_mean * points_.col(i);
        }

        return weighted_mean;
    }

protected:
    /**
     * \brief point container
     */
    PointMatrix points_;

    /**
     * \brief weight container
     */
    Weights weights_;
};

}

#endif
