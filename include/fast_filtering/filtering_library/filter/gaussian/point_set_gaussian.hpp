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
 * \date 10/21/2014
 * \author Jan Issac (jan.issac@gmail.com)
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * Max-Planck-Institute for Intelligent Systems,
 * University of Southern California
 */

#ifndef FL__FILTER__GAUSSIAN__POINT_SET_GAUSSIAN_HPP
#define FL__FILTER__GAUSSIAN__POINT_SET_GAUSSIAN_HPP

#include <map>

#include <Eigen/Dense>

#include <fast_filtering/distributions/gaussian.hpp>
#include <fast_filtering/filtering_library/exception/exception.hpp>
#include <fast_filtering/filtering_library/filter/filter_interface.hpp>

namespace fl
{

// Forward declaration
template <typename Point, int POINT_COUNT> class PointSetGaussian;

/**
 * Trait struct of a PointSetGaussian
 */
template <typename Point_, int POINT_COUNT>
struct Traits<PointSetGaussian<Point_, POINT_COUNT>>
{
    /**
     * @brief Gaussian sample type
     */
    typedef Point_ Point;

    /**
     * \brief Weight harbors the weights of a point. For each of the first two
     * moments there is a separate weight.
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
     * \typedef PointMatrix
     *
     * \brief Point container type
     *
     * \details
     * The point container type has a fixed-size dimension of a point
     * and the number of the points is statically known.
     */
    typedef Eigen::Matrix<
                typename Point::Scalar,
                Point::SizeAtCompileTime,
                POINT_COUNT
            > PointMatrix;

    /**
     * \typedef Weight
     *
     * Weight list of all points
     */
    typedef std::vector<Weight> WeightVector;

    /**
     * \typedef Base
     *
     * Base class type
     */
    typedef ff::Gaussian<Point> Base;
};

/**
 * \class PointSetGaussian
 *
 * PointSetGaussian is Gaussian distribution represented by a set of points
 * (a statistic)
 *
 * \tparam Point        Gaussian variable type
 * \tparam POINT_COUNT  Number of points representing the gaussian
 */
template <typename Point_, int POINT_COUNT = Eigen::Dynamic>
class PointSetGaussian:
        public ff::Gaussian<Point_>
{
public:
    typedef PointSetGaussian<Point_, POINT_COUNT> This;

    typedef typename Traits<This>::Base         Base;
    typedef typename Traits<This>::Point        Point;
    typedef typename Traits<This>::PointMatrix  PointMatrix;
    typedef typename Traits<This>::Weight       Weight;
    typedef typename Traits<This>::WeightVector WeightVector;

    using Base::Mean;
    using Base::Covariance;
    using Base::Precision;
    using Base::SquareRoot;
    using Base::DiagonalCovariance;
    using Base::DiagonalSquareRoot;
    using Base::DiagonalPrecision;
    using Base::SetStandard;
    using Base::Probability;
    using Base::LogProbability;
    using Base::Dimension;
    using Base::NoiseDimension;

public:
    /**
     * Creates a PointSetGaussian
     *
     * \param dimension     Sample space dimension
     * \param point_count   Number of points representing the Gaussian
     */
    PointSetGaussian(size_t point_count = POINT_COUNT > 0 ? POINT_COUNT : 0,
                     size_t dimension = Point::SizeAtCompileTime)
        : Base(dimension),
          points_(Dimension(), point_count),
          weights_(point_count)
    {
        points_.setZero();

        double weight = (point_count > 0) ? 1./double(point_count) : 0;
        std::fill(weights_.begin(), weights_.end(), Weight{weight, weight});
    }

    /**
     * \brief Overridable default constructor
     */
    virtual ~PointSetGaussian() { }

    /**
     * \return i-th point
     *
     * \param i Index of requested point
     *
     * \throw OutOfBoundsException
     */
    Point point(size_t i) const
    {
        if (is_out_of_bounds(i))
        {
            BOOST_THROW_EXCEPTION(OutOfBoundsException(i, points_.cols()));
        }

        return points_.col(i);
    }

    /**
     * \return weight of i-th point assuming both weights are the same
     *
     * \param i Index of requested point
     *
     * \throw OutOfBoundsException
     */
    double weight(size_t i)
    {
        if (is_out_of_bounds(i))
        {
            BOOST_THROW_EXCEPTION(OutOfBoundsException(i, weights_.size()));
        }

        return weights_[i].m_mean;
    }

    /**
     * \return weight of i-th point
     *
     * \param i Index of requested point
     *
     * \throw OutOfBoundsException
     */
    const Weight& weight(size_t i) const
    {
        if (is_out_of_bounds(i))
        {
            BOOST_THROW_EXCEPTION(OutOfBoundsException(i, points_.cols()));
        }

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
    const WeightVector& weights() const noexcept
    {
        return weights_;
    }

    /**
     * Sets a given point at position i
     *
     * \param i         Index of point
     * \param p         The new point
     *
     * \throw OutOfBoundsException
     */
    void point(size_t i, Point p)
    {
        if (is_out_of_bounds(i))
        {
            BOOST_THROW_EXCEPTION(OutOfBoundsException(i, points_.cols()));
        }

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
     * \throw OutOfBoundsException
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
     * \throw OutOfBoundsException
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
     * \throw OutOfBoundsException
     */
    void point(size_t i, Point p, Weight weights)
    {
        if (is_out_of_bounds(i))
        {
            BOOST_THROW_EXCEPTION(OutOfBoundsException(i, points_.cols()));
        }

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
     * \throw OutOfBoundsException
     */
    void weight(size_t i, double w)
    {
        if (is_out_of_bounds(i))
        {
            BOOST_THROW_EXCEPTION(OutOfBoundsException(i, weights_.size()));
        }

        weights_[i] = Weight{w, w};
    }

    /**
     * Sets given weights of a point at position i
     *
     * \param i         Index of point
     * \param w_mean    point weight used to compute the first moment
     * \param w_cov     point weight used to compute the second centered moment
     *
     * \throw OutOfBoundsException
     */
    void weight(size_t i, Point p, double w_mean , double w_cov)
    {
        if (is_out_of_bounds(i))
        {
            BOOST_THROW_EXCEPTION(OutOfBoundsException(i, weights_.size()));
        }

        weights_[i] = Weight{w_mean, w_cov};
    }

    /**
     * Sets given weights of a point at position i
     *
     * \param i         Index of point
     * \param weights   point weights
     *
     * \throw OutOfBoundsException
     */
    void weight(size_t i, Weight weights)
    {
        if (is_out_of_bounds(i))
        {
            BOOST_THROW_EXCEPTION(OutOfBoundsException(i, weights_.size()));
        }

        weights_[i] = weights;
    }

    /**
     * @return Centered points matrix.
     *
     * Creates a PointMatrix populated with zero mean points
     */
    PointMatrix centered_points() const noexcept
    {
        PointMatrix centered(points_.rows(), points_.cols);

        const size_t point_count = points_.cols();
        for (size_t i = 0; i < point_count; ++i)
        {
            centered.col(i) = points_.col(i) - Mean();
        }

        return centered;
    }

protected:
    /**
     * \return True of index i is out of bounds
     *
     * \param i Point index
     *
     * \internal asserts size equallity of points and weights
     */
    bool is_out_of_bounds(size_t i) const noexcept
    {
        assert(points_.cols() == weights_.size());

        return !(i < weights_.size());
    }

protected:
    /**
     * \brief point container
     */
    PointMatrix points_;

    /**
     * \brief weight container
     */
    WeightVector weights_;
};

}

#endif
