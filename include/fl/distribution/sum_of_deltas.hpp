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
 * \file sum_of_deltas.hpp
 * \date 05/25/2014
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__DISTRIBUTION__SUM_OF_locations_HPP
#define FL__DISTRIBUTION__SUM_OF_locations_HPP

#include <Eigen/Dense>

// std
#include <vector>

#include <fl/util/assertions.hpp>
#include <fl/util/traits.hpp>
#include <fl/distribution/interface/moments.hpp>

namespace fl
{

// Forward declarations
template <typename Variate> class SumOfDeltas;

/**
 * SumOfDeltas distribution traits. This trait definition contains all types
 * used internally within the distribution. Additionally, it provides the types
 * needed externally to use the SumOfDeltas.
 */
template <typename Var>
struct Traits<SumOfDeltas<Var>>
{
    enum
    {
        /**
         * \brief Gaussian dimension
         *
         * For fixed-size Point type and hence a fixed-size distrobution, the
         * \c Dimension value is greater zero. Dynamic-size distrobutions have
         * the dimension Eigen::Dynamic.
         */
        Dimension = Var::RowsAtCompileTime
    };

    /**
     * \brief Distribution variate type
     */
    typedef Var Variate;

    /**
     * \brief Internal scalar type (e.g. double, float, std::complex, etc)
     */
    typedef typename Variate::Scalar Scalar;

    /**
     * \brief Distribution second moment type
     */
    typedef Eigen::Matrix<
                Scalar,
                Var::RowsAtCompileTime,
                Var::RowsAtCompileTime
            > Covariance;

    /**
     * \brief Deltas container (Sample container representing this
     * non-parametric distribution)
     */
    typedef std::vector<Var> Locations;

    /**
     * \brief Weight vector associated with the deltas
     */
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Weights;

    /**
     * \brief Moments interface of the SumOfDeltas distribution
     */
    typedef Moments<Var, Covariance> MomentsBase;
};

/**
 * \todo missing unit tests
 * \ingroup distributions
 *
 * SumOfDeltas represents a non-parametric distribution. The distribution is
 * described by a set of deltas each assiciated with a weight.
 */
template <typename Variate>
class SumOfDeltas
        : public Traits<SumOfDeltas<Variate>>::MomentsBase
{
public:
    typedef SumOfDeltas<Variate> This;

    typedef typename Traits<This>::Scalar       Scalar;
    typedef typename Traits<This>::Covariance Covariance;
    typedef typename Traits<This>::Locations    Locations;
    typedef typename Traits<This>::Weights      Weights;

public:
    /**
     * Creates a dynamic or fixed-size SumOfDeltas.
     *
     * \param dimension Dimension of the Variate. The default is defined by the
     *                  dimension of the variable type \em Vector. If the size
     *                  of the Vector at compile time is fixed, this will be
     *                  adapted. For dynamic-sized Variable the dimension is
     *                  initialized to 0.
     */
    explicit
    SumOfDeltas(size_t dim = DimensionOf<Variate>())
    {
        locations_ = Locations(1, Variate::Zero(dim));
        weights_ = Weights::Ones(1);
    }

    /**
     * \brief Overridable default constructor
     */
    virtual ~SumOfDeltas() { }

    /**
     * Sets the distribution deltas and their weights. This overrides the entire
     * distribution.
     *
     * \param [in] deltas    The new set of deltas
     * \param [in] weights   THe weights of deltas
     */
    virtual void set_deltas(const Locations& locations, const Weights& weights)
    {
        locations_ = locations;
        weights_ = weights.normalized();
    }

    /**
     * Accesses the deltas and their weights
     *
     * \param [out] locations
     * \param [out] weights
     */
    virtual void get_deltas(Locations& locations, Weights& weights) const
    {
        locations = locations_;
        weights = weights_;
    }

    Variate& location(size_t i)
    {
        return locations_[i];
    }

    Scalar& weight(size_t i)
    {
        return weights_(i);
    }



    virtual void resize(size_t dim)
    {
        locations_.resize(dim);
        weights_.resize(dim);
    }

    virtual size_t size()
    {
        return locations_.size();
    }

    /**
     * \return The weighted mean of the locations, or simply the first moment of
     *         the distribution.
     */
    virtual Variate mean() const
    {
        Variate mu(Variate::Zero(dimension()));
        for(size_t i = 0; i < locations_.size(); i++)
            mu += weights_[i] * locations_[i];

        return mu;
    }

    /**
     * \return The covariance or the second central moment of the distribution
     */
    virtual Covariance covariance() const
    {
        Variate mu = mean();
        Covariance cov(Covariance::Zero(dimension(), dimension()));
        for(size_t i = 0; i < locations_.size(); i++)
            cov += weights_[i] * (locations_[i]-mu) * (locations_[i]-mu).transpose();

        return cov;
    }

    /**
     * \return Dimension of the distribution variate
     */
    virtual int dimension() const
    {
        return locations_[0].rows();
    }

protected:
    Locations  locations_;
    Weights weights_;
};

}

#endif
