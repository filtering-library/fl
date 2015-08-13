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
 * \file t_distribution.hpp
 * \date August 2015
 * \author Jan Issac (jan.issac@gmail.com)
 * \author Cristina Garcia Cifuentes (c.garciacifuentes@gmail.com)
 */

#ifndef FL__DISTRIBUTION__T_DISTRIBUTION_HPP
#define FL__DISTRIBUTION__T_DISTRIBUTION_HPP

#include <Eigen/Dense>

#include <random>
#include <boost/math/distributions.hpp>

#include <fl/util/meta.hpp>
#include <fl/util/types.hpp>
#include <fl/exception/exception.hpp>
#include <fl/distribution/gaussian.hpp>
#include <fl/distribution/chi_squared.hpp>
#include <fl/distribution/interface/evaluation.hpp>
#include <fl/distribution/interface/moments.hpp>
#include <fl/distribution/interface/standard_gaussian_mapping.hpp>

namespace fl
{

/**
 * \ingroup distributions
 *
 * \brief TDistribution represents a multivariate student's t-distribution
 * \f$t_\nu(\mu, \Sigma)\f$, where \f$\nu \in \mathbb{R} \f$ is the
 * degree-of-freedom, \f$\mu\in \mathbb{R}^n\f$ the distribution location and
 * \f$\Sigma \in \mathbb{R}^{n\times n} \f$ the scaling or covariance matrix.
 */
template <typename Variate>
class TDistribution
    : public Moments<Variate>,
      public Evaluation<Variate>,
      public StandardGaussianMapping<
                Variate,
                JoinSizes<SizeOf<Variate>::Value, 1>::Value>
{
private:
    typedef StandardGaussianMapping<
                Variate,
                JoinSizes<SizeOf<Variate>::Value, 1>::Value
            > StdGaussianMappingBase;

public:
    /**
     * \brief Second moment matrix type, i.e covariance matrix
     */
    typedef typename Moments<Variate>::SecondMoment SecondMoment;

    /**
     * \brief Represents the StandardGaussianMapping standard variate type which
     *        is of the same dimension as the \c TDistribution \c Variate. The
     *        StandardVariate type is used to sample from a standard normal
     *        Gaussian and map it to this \c TDistribution
     */
    typedef typename StdGaussianMappingBase::StandardVariate StandardVariate;

public:
    /**
     * Creates a dynamic or fixed size t-distribution.
     *
     * \param degrees_of_freedom
     *                  t-distribution degree-of-freedom
     * \param dimension Dimension of the distribution. The default is defined by
     *                  the dimension of the variable type \em Vector. If the
     *                  size of the Variate at compile time is fixed, this will
     *                  be adapted. For dynamic-sized Variable the dimension is
     *                  initialized to 0.
     */
    explicit TDistribution(Real degrees_of_freedom,
                           int dim = DimensionOf<Variate>())
        : StdGaussianMappingBase(dim + 1),
          chi2_(degrees_of_freedom),
          normal_(dim)
    {
        static_assert(Variate::SizeAtCompileTime != 0,
                      "Illegal static dimension");

        normal_.set_standard();
    }

    /**
     * \brief Overridable default destructor
     */
    virtual ~TDistribution() { }

    /**
     * \return a t-distribution sample of the type \c Variate determined by
     * mapping a standard normal sample into the t-distribution sample space
     *
     * \param sample    Standard normal sample
     *
     * \throws See Gaussian<Variate>::map_standard_normal
     */
    Variate map_standard_normal(const StandardVariate& sample) const override
    {
        assert(sample.size() == dimension() + 1);

        Real u = chi2_.map_standard_normal(sample.bottomRows(1)(0));
        Variate n = normal_.map_standard_normal(sample.topRows(dimension()));

        // rvo
        Variate v = location() + std::sqrt(degrees_of_freedom() / u) * n;
        return v;
    }

    /**
     * \return Log of the probability of the given sample \c variate
     *
     * Evaluates the t-distribution pdf
     *
     *  \f$ t_\nu(\mu, \Sigma) = \frac{\Gamma\left[(\nu+p)/2\right]}
     *          {\Gamma(\nu/2)
     *           \nu^{p/2}\pi^{p/2}
     *           \left|{\boldsymbol\Sigma}\right|^{1/2}
     *           \left[1 +
     *                 \frac{1}{\nu}
     *                 ({\mathbf x}-{\boldsymbol\mu})^T
     *                 {\boldsymbol\Sigma}^{-1}
     *                 ({\mathbf x}-{\boldsymbol\mu})\right]^{(\nu+p)/2}} \f$
     *
     * at location \f${\mathbf x}\f$
     *
     *
     * \param variate sample which should be evaluated
     *
     * \throws See Gaussian<Variate>::has_full_rank()
     */
    Real log_probability(const Variate& x) const override
    {
        return cached_log_pdf_.log_probability(*this, x);
    }

    /**
     * \return Gaussian dimension
     */
    virtual constexpr int dimension() const
    {
        return normal_.dimension();
    }

    /**
     * \return t-distribution location
     */
    const Variate& mean() const override
    {
        return location();
    }

    /**
     * \return t-distribution scaling matrix
     *
     * \throws See Gaussian<Variate>::covariance()
     */
    const SecondMoment& covariance() const override
    {
        return normal_.covariance();
    }

    /**
     * \return t-distribution location
     */
    virtual const Variate& location() const
    {
        return normal_.mean();
    }

    /**
     * \return t-distribution degree-of-freedom
     */
    virtual Real degrees_of_freedom() const
    {
        return chi2_.degrees_of_freedom();
    }

    /**
     * Changes the dimension of the dynamic-size t-distribution and sets it to a
     * standard distribution with zero mean and identity covariance.
     *
     * \param new_dimension New dimension of the t-distribution
     *
     * \throws ResizingFixedSizeEntityException
     *         see standard_variate_dimension(int)
     */
    virtual void dimension(int new_dimension)
    {
        StdGaussianMappingBase::standard_variate_dimension(new_dimension + 1);
        normal_.dimension(new_dimension);
        cached_log_pdf_.flag_dirty();
    }

    /**
     * Sets the distribution location
     *
     * \param location New t-distribution mean
     *
     * \throws WrongSizeException
     */
    virtual void location(const Variate& new_location) noexcept
    {
        normal_.mean(new_location);
        cached_log_pdf_.flag_dirty();
    }

    /**
     * Sets the covariance matrix
     * \param covariance New covariance matrix
     *
     * \throws WrongSizeException
     */

    virtual void scaling_matrix(const SecondMoment& scaling_matrix)
    {
        normal_.covariance(scaling_matrix);
        cached_log_pdf_.flag_dirty();
    }

    /**
     * Sets t-distribution degree-of-freedom
     */
    virtual void degrees_of_freedom(Real dof)
    {
        chi2_.degrees_of_freedom(dof);
        cached_log_pdf_.flag_dirty();
    }

protected:
    /** \cond internal */
    ChiSquared chi2_;
    Gaussian<Variate> normal_;
    /** \endcond */


private:
    class CachedLogPdf
    {
    public:
        CachedLogPdf()
            : dirty_(true)
        { }

        /**
         * Evaluates the t-distribution pdf at a given position \c x
         */
        Real log_probability(
                const TDistribution<Variate>& t_distr, const Variate& x)
        {
            if (dirty_) update(t_distr);

            Variate z = x - t_distr.location();
            Real dof = t_distr.degrees_of_freedom();

            Real quad_term = (z.transpose() * t_distr.normal_.precision() * z);
            Real ln_term = std::log(Real(1) + quad_term  / dof);

            return const_term_ - const_factor_ * ln_term;
        }

        void flag_dirty() { dirty_ = true; }

    private:
        void update(const TDistribution<Variate>& t_distr)
        {
            Real half = Real(1)/Real(2);
            Real dim = t_distr.dimension();
            Real dof = t_distr.degrees_of_freedom();
            const_term_ =
                boost::math::lgamma(half * (dof + dim))
                - boost::math::lgamma(half * dof)
                - half * (dim * std::log(M_PI * dof))
                - half * (std::log(t_distr.normal_.covariance_determinant()));

            const_factor_ = half * (dof + dim);

            dirty_ = false;
        }

        bool dirty_;
        Real const_factor_;
        Real const_term_;
    };

    friend class CachedLogPdf;

    mutable CachedLogPdf cached_log_pdf_;
};

}

#endif
