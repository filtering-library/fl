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

#ifndef FL__DISTRIBUTION__SUM_OF_DELTAS_HPP
#define FL__DISTRIBUTION__SUM_OF_DELTAS_HPP

#include <Eigen/Core>

// std
#include <vector>

#include <fl/util/assertions.hpp>
#include <fl/util/traits.hpp>
#include <fl/distribution/interface/moments.hpp>
#include <fl/distribution/interface/standard_gaussian_mapping.hpp>

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

    typedef Var Variate;
    typedef Eigen::Matrix<double, Dimension, 1> Mean;
    typedef Eigen::Matrix<double, Dimension, Dimension> Covariance;

    typedef std::vector<Variate> Locations;
    typedef Eigen::Array<double, Eigen::Dynamic, 1> Probabilities;

    typedef Moments<Mean, Covariance> MomentsBase;
    typedef StandardGaussianMapping<Variate, double> GaussianMappingBase;
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
        : public Traits<SumOfDeltas<Variate>>::MomentsBase,
          public Traits<SumOfDeltas<Variate>>::GaussianMappingBase
{
public:
    typedef SumOfDeltas<Variate> This;

    typedef typename Traits<This>::Mean Mean;
    typedef typename Traits<This>::Covariance Covariance;
    typedef typename Traits<This>::Locations    Locations;
    typedef typename Traits<This>::Probabilities      Probabilities;


public:
    // constructor and destructor ----------------------------------------------
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
        log_probabilities_ = Probabilities::Zero(1);
        cumulative_ = std::vector<double>(1,1);
    }

    virtual ~SumOfDeltas() { }




    /// non-const functions ****************************************************

    // set ---------------------------------------------------------------------
    virtual void log_unnormalized_probabilities(const Probabilities& log_probs)
    {
        // rescale for numeric stability
        log_probabilities_ = log_probs;
        set_max(log_probabilities_);
\
        // copy to probabilities
        probabilities_ = log_probabilities_.exp();
        double sum = probabilities_.sum();

        // normalize
        probabilities_ /= sum;
        log_probabilities_ -= std::log(sum);

        // compute cumulative
        cumulative_.resize(log_probs.size());
        cumulative_[0] = probabilities_[0];
        for(size_t i = 1; i < cumulative_.size(); i++)
            cumulative_[i] = cumulative_[i-1] + probabilities_[i];

        // resize locations
        locations_.resize(log_probs.size());
    }    

    virtual Variate& location(size_t i)
    {
        return locations_[i];
    }

    virtual void resize(size_t dim)
    {
        locations_.resize(dim);
        log_probabilities_.resize(dim);
    }


    /// const functions ********************************************************
     
    // sampling ----------------------------------------------------------------
    virtual Variate map_standard_normal(const double& gaussian_sample) const
    {
        double uniform_sample =
                0.5 * (1.0 + std::erf(gaussian_sample / std::sqrt(2.0)));

        return map_standard_uniform(uniform_sample);
    }

    virtual Variate map_standard_uniform(const double& uniform_sample) const
    {
        typename std::vector<double>::const_iterator
                iterator = std::lower_bound(cumulative_.begin(),
                                            cumulative_.end(),
                                            uniform_sample);

        int index = iterator - cumulative_.begin();
        return locations_[index];
    }


    // get ---------------------------------------------------------------------
    virtual double log_probability(const size_t& i) const
    {
        return log_probabilities_(i);
    }

    virtual double probability(const size_t& i) const
    {
        return std::exp(log_probabilities_(i));
    }

    virtual size_t size() const
    {
        return locations_.size();
    }

    virtual int dimension() const
    {
        return locations_[0].rows();
    }


    // compute properties ------------------------------------------------------
    virtual Mean mean() const
    {
        Mean mu(Mean::Zero(dimension()));
        for(size_t i = 0; i < locations_.size(); i++)
            mu += probability(i) * locations_[i].template cast<double>();

        return mu;
    }

    virtual Covariance covariance() const
    {
        Mean mu = mean();
        Covariance cov(Covariance::Zero(dimension(), dimension()));
        for(size_t i = 0; i < locations_.size(); i++)
        {
            Mean delta = (locations_[i].template cast<double>()-mu);
            cov += probability(i) * delta * delta.transpose();
        }

        return cov;
    }

    virtual double entropy() const
    {
        double ent = 0;
        for(int i = 0; i < log_probabilities_.size(); i++)
        {
            double summand =
                    - log_probabilities_(i) * std::exp(log_probabilities_(i));

            if(!std::isfinite(summand))
                summand = 0; // the limit for weight -> 0 is equal to 0
            ent += summand;
        }

        return ent;
    }





protected:
    virtual void set_max(Probabilities& p, const double& max = 0) const
    {
        const double old_max = p.maxCoeff();
        p += max - old_max;
    }


protected:
    Locations  locations_;

    Probabilities log_probabilities_;
    Probabilities probabilities_;
    std::vector<double> cumulative_;
};

}

#endif
