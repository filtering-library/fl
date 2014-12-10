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
 * Max-Planck-Institute for Intelligent Systems,
 * University of Southern California
 */

#ifndef FL__FILTER__GAUSSIAN__POINT_SET_GAUSSIAN_HPP
#define FL__FILTER__GAUSSIAN__POINT_SET_GAUSSIAN_HPP

#include <map>
#include <cmath>
#include <type_traits>

#include <Eigen/Dense>

#include <fast_filtering/distributions/gaussian.hpp>
#include <fast_filtering/filtering_library/exception/exception.hpp>
#include <fast_filtering/filtering_library/filter/filter_interface.hpp>

namespace fl
{

// Forward declaration
template <typename Point, int Points_> class PointSetGaussian;

/**
 * Trait struct of a PointSetGaussian
 */
template <typename Point_, int Points_>
struct Traits<PointSetGaussian<Point_, Points_>>
{    
    /**
     * \brief Gaussian sample type
     */
    typedef Point_ Point;

    /**
     * \brief Base class type
     */
    typedef ff::Gaussian<Point> Base;

    /* Gaussian Base types */
    typedef typename ff::Traits<Base>::Vector   Vector;
    typedef typename ff::Traits<Base>::Scalar   Scalar;
    typedef typename ff::Traits<Base>::Noise    Noise;
    typedef typename ff::Traits<Base>::Operator Operator;

    /**
     * \brief Number of points for fixed-size set
     */
    enum
    {
        /**
         * \brief Gaussian dimension
         *
         * For fixed-size Point type and hence a fixed-size Gaussian, the
         * \c Dimension value is greater zero. Dynamic-size Gaussians have the
         * dymension Eigen::Dynamic.
         */
        Dimension = ff::Traits<Base>::Dimension,

        /**
         * \brief Number of points which provided by the PointSetGaussian.
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
    typedef Eigen::Matrix<Scalar, Dimension, Points_ > PointMatrix;

    /**
     * @brief WeightVector
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
 * \class PointSetGaussian
 *
 * PointSetGaussian is Gaussian distribution represented by a set of points
 * (a statistic)
 *
 * \tparam Point    Gaussian variable type
 * \tparam Points_  Number of points representing the gaussian
 */
template <typename Point_, int Points_ = Eigen::Dynamic>
class PointSetGaussian:
        public ff::Gaussian<Point_>
{
public:
    typedef PointSetGaussian<Point_, Points_> This;

    typedef typename Traits<This>::Base         Base;
    typedef typename Traits<This>::Scalar       Scalar;
    typedef typename Traits<This>::Operator     Operator;
    typedef typename Traits<This>::Noise        Noise;
    typedef typename Traits<This>::Point        Point;
    typedef typename Traits<This>::PointMatrix  PointMatrix;
    typedef typename Traits<This>::Weight       Weight;
    typedef typename Traits<This>::Weights      Weights;
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
     * \param points_count   Number of points representing the Gaussian
     * \param dimension      Sample space dimension
     */
    PointSetGaussian(size_t points_count = (Points_ > 0) ? Points_ : 0,
                     size_t dimension = Point::RowsAtCompileTime)
        : Base(dimension),
          points_(Dimension(), points_count),
          weights_(points_count),
          changes_staged_(false)
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
    virtual ~PointSetGaussian() { }

    /**
     * Resizes a dynamic-size PointSetGaussian to a desired. For
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

        points_.setZero(Dimension(), points_count);

        weights_.resize(points_count);
        const double weight = (points_count > 0)? 1. / double(points_count) : 0;
        std::fill(weights_.begin(), weights_.end(), Weight{weight, weight});
    }

    /**
     * Sets the new Gaussian dimension for dynamic-size Gaussian (not to confuse
     * with the number of points)
     *
     * @param new_dimension New Gaussian dimension
     */
    virtual void Dimension(size_t new_dimension)
    {
        Base::Dimension(new_dimension);
        points_.resize(Dimension(), count_points());
    }

    /**
     * @return The number of points
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
        assert_bounds(i);

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
        assert_bounds(i);

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
        assert_bounds(i);

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
     * @return Returns the weights for the mean of the points as a vector
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
     * @return Returns the weights for the covariance of the points as a vector
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
        assert_bounds(i);

        points_.col(i) = p;
        stage_point_changes();
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
        assert_bounds(i);

        points_.col(i) = p;
        weights_[i] = weights;
        stage_point_changes();
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
        assert_bounds(i);

        weights_[i] = weights;
        stage_point_changes();
    }

    void commit_point_changes()
    {
        changes_staged_ = false;

        if (moments_dirty_)
        {
            PointMatrix P = weighted_centered_points();
            Covariance(P * P.transpose());
            Mean(weighted_mean());

            moments_dirty_ = false;
        }
    }

    /**
     * @return Centered points matrix.
     *
     * Creates a PointMatrix populated with zero mean points
     */
    PointMatrix centered_points() const noexcept
    {
        PointMatrix centered(points_.rows(), points_.cols());

        const Point weighted_mean = weighted_mean();
        const size_t point_count = points_.cols();
        for (size_t i = 0; i < point_count; ++i)
        {
            centered.col(i) = points_.col(i) - weighted_mean;
        }

        return centered;
    }

    /**
     * @return Weighted centered points matrix.
     *
     * Creates a PointMatrix populated with weighted zero mean points
     */
    PointMatrix weighted_centered_points() const noexcept
    {
        PointMatrix centered(points_.rows(), points_.cols());

        const Point mean = weighted_mean();
        const size_t point_count = points_.cols();
        for (size_t i = 0; i < point_count; ++i)
        {
            centered.col(i) =
                    weights_[i].w_cov * (points_.col(i) - mean);
        }

        return centered;
    }

    /**
     * @return The weighted mean of all points
     */
    Point weighted_mean() const
    {
        Point weighted_mean;
        weighted_mean.setZero(Dimension());

        const size_t point_count = points_.cols();
        for (size_t i = 0; i < point_count; ++i)
        {
            weighted_mean += weights_[i].w_mean * points_.col(i);
        }

        return weighted_mean;
    }

    /**
     * The point set becomes dirty once one of moment representations has been
     * changed.
     *
     * @return point set dirty status
     */
    bool is_point_set_dirty() const noexcept
    {
        return point_set_dirty_;
    }

    /* Gaussian interface */

    virtual Vector Mean() const
    {
        if (moments_dirty_)
        {

        }

        return mean_;
    }

    virtual void Mean(const Vector& mean) noexcept
    {
        point_set_dirty_ = true;
        Base::Mean(mean);
    }

    virtual void Covariance(const Operator& covariance) noexcept
    {
        point_set_dirty_ = true;
        Covariance(covariance);
    }

    virtual void SquareRoot(const Operator& square_root) noexcept
    {
        point_set_dirty_ = true;
        SquareRoot(square_root);
    }

    virtual void Precision(const Operator& precision) noexcept
    {
        point_set_dirty_ = true;
        Precision(precision);
    }

    virtual void DiagonalCovariance(const Operator& diag_covariance) noexcept
    {
        point_set_dirty_ = true;
        DiagonalCovariance(diag_covariance);
    }

    virtual void DiagonalSquareRoot(const Operator& diag_square_root) noexcept
    {
        point_set_dirty_ = true;
        DiagonalSquareRoot(diag_square_root);
    }

    virtual void DiagonalPrecision(const Operator& diag_precision) noexcept
    {
        point_set_dirty_ = true;
        DiagonalPrecision(diag_precision);
    }

protected:
    /**
     * Asserts that the specified index is within bounds and that the dimension
     * is greater zero.
     *
     * \param i Point index
     *
     * \internal asserts size equallity of points and weights
     *
     * \throws OutOfBoundsException
     * \throws ZeroDimensionException
     */
    void assert_bounds(size_t i) const
    {
        assert(points_.cols() == weights_.size());

        if (i >= weights_.size())
        {
            BOOST_THROW_EXCEPTION(OutOfBoundsException(i, weights_.size()));
        }

        if (Dimension() == 0)
        {
            BOOST_THROW_EXCEPTION(ZeroDimensionException("PointSetGaussian"));
        }
    }


    void stage_point_changes()
    {
        moments_dirty_ = true;
        changes_staged_ = true;
    }

protected:
    /**
     * \brief point container
     */
    mutable PointMatrix points_;

    /**
     * \brief weight container
     */
    mutable Weights weights_;

    bool changes_staged_;
    bool moments_dirty_;
    bool point_set_dirty_;
};

}

#endif
