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
 * \file joint_distribution_id.hpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__DISTRIBUTION__JOINT_DISTRIBUTION_ID_HPP
#define FL__DISTRIBUTION__JOINT_DISTRIBUTION_ID_HPP

#include <Eigen/Dense>

#include <fl/util/meta.hpp>
#include <fl/util/traits.hpp>

#include <fl/distribution/interface/moments.hpp>

namespace fl
{

// Forward declaration
template <typename...Distribution> class JointDistribution;

/**
 * \internal
 *
 * Traits of JointDistribution<Distribution...>
 */
template <typename...Distribution>
struct Traits<JointDistribution<Distribution...>>
{
    enum : signed int
    {
        JointSize = JoinSizes<SizeOf<typename Distribution::Variate>()...>::Size
    };

    typedef typename FirstTypeIn<
                typename Distribution::Variate...
            >::Type::Scalar Scalar;

    typedef Eigen::Matrix<Scalar, JointSize, 1> Variate;
};

/**
 * \ingroup distributions
 */
template <typename...Distribution>
class JointDistribution
    : public Moments<Traits<JointDistribution<Distribution...>>::Variate>
{
public:
    typedef Traits<JointDistribution<Distribution...>>::Variate Variate;
    typedef typename Moments<Variate>::SecondMoment SecondMoment;
    typedef std::tuple<Distribution...> MarginalDistributions;

public:
    JointDistribution(Distribution...distributions)
        : distributions_(distributions...)
    { }

    /**
     * \brief Overridable default destructor
     */
    virtual ~JointDistribution() { }

    virtual Variate mean() const
    {
        Variate mu = Variate(dimension(), 1);

        mean_<sizeof...(Distribution)>(distributions_, mu);

        return mu;
    }

    virtual SecondMoment covariance() const
    {
        SecondMoment cov = SecondMoment::Zero(dimension(), dimension());

        covariance<sizeof...(Distribution)>(distributions_, cov);

        return cov;
    }

    virtual int dimension() const
    {
        return expend_dimension(CreateIndexSequence<sizeof...(Distribution)>());
    }

    MarginalDistributions& distributions()
    {
        return distributions_;
    }

    const MarginalDistributions& distributions() const
    {
        return distributions_;
    }

protected:
    MarginalDistributions distributions_;

private:
    template <int...Indices>
    int expend_dimension(IndexSequence<Indices...>) const
    {
        const auto& dims = { std::get<Indices>(distributions_).dimension()... };

        int joint_dim = 0;
        for (auto dim : dims) { joint_dim += dim; }

        return joint_dim;
    }

    template <int Size, int k = 0>
    void mean_(const MarginalDistributions& distr_tuple,
               Variate& mu,
               int offset = 0) const
    {
        auto&& distribution = std::get<k>(distr_tuple);
        const int dim = distribution.dimension();

        mu.middleRows(offset, dim) = distribution.mean();

        if (Size == k + 1) return;

        mean_<Size, k + (k + 1 < Size ? 1 : 0)>(distr_tuple, mu, offset + dim);
    }

    template <int Size, int k = 0>
    void covariance(const MarginalDistributions& distr,
                    SecondMoment& cov,
                    const int offset = 0) const
    {
        auto& distribution = std::get<k>(distr);
        const int dim = distribution.dimension();

        cov.block(offset, offset, dim, dim) = distribution.covariance();

        if (Size == k + 1) return;

        covariance<Size, k + (k + 1 < Size ? 1 : 0)>(distr, cov, offset + dim);
    }
};

}

#endif
