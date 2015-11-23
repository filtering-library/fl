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
 * \date 2014
 * \author Jan Issac (jan.issac@gmail.com)
 * Max-Planck-Institute for Intelligent Systems,
 * University of Southern California
 */

#include <gtest/gtest.h>

#include <memory>

#include <fl/util/traits.hpp>
#include <fl/exception/exception.hpp>
#include <fl/filter/filter_interface.hpp>

/*
 * FilterForFun forward declaration
 */
class FilterForFun;

namespace fl
{
/*
 * FilterForFun Traits
 */
template <> struct Traits<FilterForFun>
{
    typedef double State;
    typedef double Input;
    typedef double Obsrv;
    typedef double Belief;
};
}

/*
 * FilterForFun Stub Filter
 */
class FilterForFun:
        public fl::FilterInterface<FilterForFun>
{
public:
    typedef FilterForFun This;

    typedef typename fl::Traits<This>::State State;
    typedef typename fl::Traits<This>::Input Input;
    typedef typename fl::Traits<This>::Obsrv Obsrv;
    typedef typename fl::Traits<This>::Belief Belief;

    virtual void predict(const Belief& prior_belief,
                         const Input& input,
                         Belief& predicted_belief) override
    {
        predicted_belief = (prior_belief * 2);
    }

    virtual void update(const Belief& predicted_belief,
                        const Obsrv& observation,
                        Belief& posterior_belief) override
    {
        posterior_belief = (predicted_belief + observation) / 2.;
    }

    std::string name() const override
    {
        return "FilterForMoreFun";
    }

    std::string description() const override
    {
        return "FilterForMoreFun";
    }
};


// == FilterForMoreFun Stub Filter ========================================== //

/*
 * FilterForMoreFun forward declaration
 */
template <typename A, typename B, typename C> class FilterForMoreFun;

namespace fl
{
/*
 * FilterForMoreFun Traits
 */
template <typename A, typename B, typename C>
struct Traits<FilterForMoreFun<A, B, C>>
{
    typedef double State;
    typedef double Input;
    typedef double Obsrv;
    typedef double Belief;
};
}

/*
 * FilterForMoreFun Stub Filter
 */
template <typename A, typename B, typename C>
class FilterForMoreFun:
        public fl::FilterInterface<FilterForMoreFun<A, B, C>>
{
public:
    typedef FilterForMoreFun<A, B, C> This;

    typedef typename fl::Traits<This>::State State;
    typedef typename fl::Traits<This>::Input Input;
    typedef typename fl::Traits<This>::Obsrv Obsrv;
    typedef typename fl::Traits<This>::Belief Belief;

    void predict(const Belief& prior_belief,
                         const Input& input,
                         Belief& predicted_belief) override
    {
        predicted_belief = (prior_belief * 3);
    }

    void update(const Belief& predicted_belief,
                        const Obsrv& observation,
                        Belief& posterior_belief) override
    {
        posterior_belief = (predicted_belief + observation) / 3.;
    }

    std::string name() const override
    {
        return "FilterForMoreFun";
    }

    std::string description() const override
    {
        return "FilterForMoreFun";
    }
};
