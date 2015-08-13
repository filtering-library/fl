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
 * \file chi_squared.hpp
 * \date August 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__DISTRIBUTION__CHI_SQUARED_HPP
#define FL__DISTRIBUTION__CHI_SQUARED_HPP

#include <Eigen/Dense>

#include <boost/math/distributions.hpp>

#include <fl/util/meta.hpp>
#include <fl/util/scalar_matrix.hpp>
#include <fl/distribution/uniform_distribution.hpp>
#include <fl/distribution/interface/evaluation.hpp>
#include <fl/distribution/interface/standard_gaussian_mapping.hpp>

namespace fl
{

/**
 * \ingroup distributions
 *
 * \brief ChiSquared represents a univariate Chi-squared distribution
 * \f$\chi^2_k\f$, with \f$k \in \mathbb{N}^{*}\f$ degrees-of-freedom
 */
class ChiSquared
    : public Evaluation<ScalarMatrix>,
      public StandardGaussianMapping<ScalarMatrix, 1>
{
private:
    typedef StandardGaussianMapping<ScalarMatrix, 1> StdGaussianMappingBase;

public:
    /**
     * \brief Represents the StandardGaussianMapping standard variate type which
     *        is of the same dimension as the \c TDistribution \c Variate. The
     *        StandardVariate type is used to sample from a standard normal
     *        Gaussian and map it to this \c TDistribution
     */
    typedef ScalarMatrix Variate;
    typedef typename StdGaussianMappingBase::StandardVariate StandardVariate;

public:
    /**
     * Creates a dynamic or fixed size t-distribution.
     *
     * \param degrees_of_freedom
     *                  t-distribution degree-of-freedom
     * \param dimension Dimension of the distribution. The default is defined by
     *                  the dimension of the variable type \em Vector. If the
     *                  size of the Vector at compile time is fixed, this will
     *                  be adapted. For dynamic-sized Variable the dimension is
     *                  initialized to 0.
     */
    explicit ChiSquared(Real degrees_of_freedom)
       : StdGaussianMappingBase(1),
        chi2_(degrees_of_freedom)
    { }

    /**
     * \brief Overridable default destructor
     */
    virtual ~ChiSquared() { }

    /**
     * \return a t-distribution sample of the type \c Variate determined by
     * mapping a standard normal sample into the t-distribution sample space
     *
     * \param n    Standard normal sample
     *
     * \throws See Gaussian<Variate>::map_standard_normal
     */
    virtual Variate map_standard_uniform(const StandardVariate& n) const
    {
        return boost::math::quantile(chi2_, n);
    }

    /**
     * \return a t-distribution sample of the type \c Variate determined by
     * mapping a standard normal sample into the t-distribution sample space
     *
     * \param n    Standard normal sample
     *
     * \throws See Gaussian<Variate>::map_standard_normal
     */
    Variate map_standard_normal(const StandardVariate& n) const override
    {
        return map_standard_uniform(uniform_.map_standard_normal(n));
    }

    /**
     * \return Log of the probability of the given sample \c variate
     *
     * \param variate sample which should be evaluated
     *
     * \throws See Gaussian<Variate>::has_full_rank()
     */
    Real log_probability(const Variate& variate) const override
    {
        assert(variate.size() == 1);
        return std::log(probability(variate));
    }

    /**
     * Determines the probability for the specified variate.
     *
     * \param variate Sample \f$x\f$ to evaluate
     *
     * \return \f$p(x)\f$
     */
    Real probability(const Variate& variate) const override
    {
        assert(variate.size() == 1);
        return boost::math::pdf(chi2_, variate);
    }

    /**
     * \return t-distribution degree-of-freedom
     */
    Real degrees_of_freedom() const
    {
        return chi2_.degrees_of_freedom();
    }

    /**
     * Sets t-distribution degree-of-freedom
     */
    void degrees_of_freedom(Real dof)
    {
        chi2_ = boost::math::chi_squared_distribution<Real>(dof);
    }

protected:
    /** \cond internal */
    UniformDistribution uniform_;
    boost::math::chi_squared_distribution<Real> chi2_;
    /** \endcond */
};

}

#endif
