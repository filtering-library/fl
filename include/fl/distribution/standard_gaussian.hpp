/*
 * This is part of the fl library, a C++ Bayesian filtering library
 * (https://github.com/filtering-library)
 *
 * Copyright (c) 2015 Max Planck Society,
 * 				 Autonomous Motion Department,
 * 			     Institute for Intelligent Systems
 *
 * This Source Code Form is subject to the terms of the MIT License (MIT).
 * A copy of the license can be found in the LICENSE file distributed with this
 * source code.
 */

/**
 * \file standard_gaussian.hpp
 * \date May 2014
 * \author Jan Issac (jan.issac@gmail.com)
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#pragma once


#include <Eigen/Dense>

#include <random>
#include <type_traits>

#include <fl/util/math.hpp>
#include <fl/util/random.hpp>
#include <fl/util/traits.hpp>
#include <fl/util/types.hpp>
#include <fl/distribution/interface/sampling.hpp>
#include <fl/distribution/interface/moments.hpp>
#include <fl/exception/exception.hpp>

namespace fl
{

/**
 * \ingroup distributions
 */
template <typename StandardVariate>
class StandardGaussian
    : public Sampling<StandardVariate>,
      public Moments<
                StandardVariate,
                typename DiagonalSecondMomentOf<StandardVariate>::Type>
{
public:
    typedef StandardVariate Variate;

    typedef Moments<
                StandardVariate,
                typename DiagonalSecondMomentOf<StandardVariate>::Type
            > MomentsBase;

    typedef typename MomentsBase::SecondMoment SecondMoment;
    typedef typename MomentsBase::SecondMoment DiagonalSecondMoment;

public:
    explicit
    StandardGaussian(int dim = DimensionOf<StandardVariate>())
        : dimension_ (dim),
          mu_(Variate::Zero(dim, 1)),
          cov_(DiagonalSecondMoment(dim)),
          generator_(fl::seed()),
          gaussian_distribution_(0.0, 1.0)
    {
        cov_.setIdentity();
    }

    virtual ~StandardGaussian() noexcept { }

    virtual StandardVariate sample() const
    {
        StandardVariate gaussian_sample(dimension(), 1);

        for (int i = 0; i < dimension_; i++)
        {
            gaussian_sample(i, 0) = gaussian_distribution_(generator_);
        }

        return gaussian_sample;
    }

    virtual int dimension() const
    {
        return dimension_;
    }

    virtual void dimension(int new_dimension)
    {
        if (dimension_ == new_dimension) return;

        if (fl::IsFixed<StandardVariate::SizeAtCompileTime>())
        {
            fl_throw(
                fl::ResizingFixedSizeEntityException(dimension_,
                                                     new_dimension,
                                                     "Gaussian"));
        }

        dimension_ = new_dimension;
    }

    virtual const Variate& mean() const
    {
        return mu_;
    }

    virtual const DiagonalSecondMoment& covariance() const
    {
        return cov_;
    }

protected:
    /** \cond internal */
    int dimension_;
    Variate mu_;
    DiagonalSecondMoment cov_;
    mutable fl::mt11213b generator_;
    mutable std::normal_distribution<Real> gaussian_distribution_;
    /** \endcond */
};

/**
 * Floating point implementation for Scalar types float, double and long double
 */
template <>
class StandardGaussian<Real>
    : public Sampling<Real>,
      public Moments<Real, Real>
{
public:
    StandardGaussian()
        : mu_(0.),
          var_(1.),
          generator_(fl::seed()),
          gaussian_distribution_(mu_, var_)
    { }

    Real sample() const
    {
        return gaussian_distribution_(generator_);
    }

    virtual const Real& mean() const
    {
        return mu_;
    }

    virtual const Real& covariance() const
    {
        return var_;
    }

protected:
    /** \cond internal */
    Real mu_;
    Real var_;
    mutable fl::mt11213b generator_;
    mutable std::normal_distribution<Real> gaussian_distribution_;
    /** \endcond */
};

}
