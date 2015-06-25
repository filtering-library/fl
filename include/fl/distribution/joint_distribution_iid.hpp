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
 * \file joint_distribution_iid.hpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__DISTRIBUTION__JOINT_DISTRIBUTION_IID_HPP
#define FL__DISTRIBUTION__JOINT_DISTRIBUTION_IID_HPP

#include <Eigen/Dense>

#include <fl/util/meta.hpp>
#include <fl/util/traits.hpp>

#include <fl/distribution/interface/moments.hpp>

namespace fl
{

// Forward declaration
template <typename...Distributions> class JointDistribution;

/**
 * \internal
 * Traits of JointDistribution<Distribution, Count>
 */
template <typename Distribution, int Count>
struct Traits<JointDistribution<MultipleOf<Distribution, Count>>>
{
    typedef typename Distribution::Variate MarginalVariate;

    enum : signed int
    {
        MarginalCount = Count,
        JointSize = ExpandSizes<SizeOf<MarginalVariate>(), Count>::Size
    };

    typedef typename MarginalVariate::Scalar Scalar;

    typedef Eigen::Matrix<Scalar, JointSize, 1> Variate;
};

/**
 * \ingroup distributions
 */
template <typename MarginalDistribution, int Count>
class JointDistribution<MultipleOf<MarginalDistribution, Count>>
    : public Moments<
                typename Traits<
                    JointDistribution<MultipleOf<MarginalDistribution, Count>>
                >::Variate>
{
public:
    /** Typdef of \c This for #from_traits(TypeName) helper */
    typedef JointDistribution This;

    typedef typename Traits<This>::Variate Variate;
    typedef typename Moments<Variate>::SecondMoment SecondMoment;
    typedef Eigen::Array<MarginalDistribution, Count, 1> MarginalDistributions;

public:
    explicit
    JointDistribution(MarginalDistribution marginal,
                      int count = ToDimension<Count>::Value)
        : distributions_(MarginalDistributions(count, 1))
    {
        assert(count > 0);

        for (int i = 0; i < distributions_.rows(); ++i)
        {
            distributions_(i) = marginal;
        }

        dimension_ = marginal.dimension() * count;
    }

    /**
     * \brief Overridable default destructor
     */
    virtual ~JointDistribution() { }

    virtual Variate mean() const
    {
        Variate mu = Variate(dimension(), 1);

        int offset = 0;
        for (int i = 0; i < distributions_.size(); ++i)
        {
            const MarginalDistribution& marginal = distributions_(i);
            int dim =  marginal.dimension();

            mu.middleRows(offset, dim) = marginal.mean();

            offset += dim;
        }

        return mu;
    }

    virtual SecondMoment covariance() const
    {
        SecondMoment cov = SecondMoment::Zero(dimension(), dimension());

        int offset = 0;
        for (int i = 0; i < distributions_.size(); ++i)
        {
            const MarginalDistribution& marginal = distributions_(i);
            int dim =  marginal.dimension();

            cov.block(offset, offset, dim, dim) = marginal.covariance();

            offset += dim;
        }

        return cov;
    }

    virtual int dimension() const
    {
        return dimension_;
    }

    MarginalDistributions& distributions()
    {
        return distributions_;
    }

    const MarginalDistributions& distributions() const
    {
        return distributions_;
    }

    MarginalDistribution& distribution(int index)
    {
        assert(index < distributions_.size());
        return distributions_(index);
    }

    const MarginalDistribution& distribution(int index) const
    {
        assert(index < distributions_.size());
        return distributions_(index);
    }

protected:
    MarginalDistributions distributions_;
    int dimension_;
};

}

#endif
