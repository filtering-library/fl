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
 * \file feature_policy.hpp
 * \date March 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__FILTER__GAUSSIAN__FEATURE_POLICY_HPP
#define FL__FILTER__GAUSSIAN__FEATURE_POLICY_HPP

#include <fl/util/traits.hpp>
#include <fl/util/meta.hpp>

namespace fl
{

template <typename Obsrv, typename Feature> class FeaturePolicyInterface;

template <typename Obsrv, typename Feature>
class FeaturePolicyInterface
{
public:
//    virtual Feature extract(const Obsrv& obsrv,
//                            const Obsrv& expected_obsrv) = 0;
};


template <typename ...> class IdentityFeaturePolicy { };

template <typename Obsrv>
struct Traits<IdentityFeaturePolicy<Obsrv>>
{
    typedef Obsrv ObsrvFeature;
    typedef FeaturePolicyInterface<Obsrv, ObsrvFeature> FeaturePolicyBase;
};

template <typename Obsrv>
class IdentityFeaturePolicy<Obsrv>
    : public Traits<IdentityFeaturePolicy<Obsrv>>::FeaturePolicyBase
{
public:
    typedef IdentityFeaturePolicy This;
    typedef from_traits(ObsrvFeature);

    virtual ObsrvFeature extract(const Obsrv& obsrv,
                                 const Obsrv& expected_obsrv,
                                 const Obsrv& var_obsrv)
    {
        ObsrvFeature feature = obsrv; // RVO
        return feature;
    }

    static constexpr int feature_dimension(int obsrv_dimension)
    {
        return obsrv_dimension;
    }
};

}

#endif
