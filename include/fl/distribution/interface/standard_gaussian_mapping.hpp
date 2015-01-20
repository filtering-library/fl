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
 * \file standard_gaussian_mapping.hpp
 * \date May 2014
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__DISTRIBUTION__INTERFACE__GAUSSIAN_MAP_HPP
#define FL__DISTRIBUTION__INTERFACE__GAUSSIAN_MAP_HPP

#include <Eigen/Dense>

#include <type_traits>

#include <fl/util/traits.hpp>
#include <fl/distribution/interface/sampling.hpp>
#include <fl/distribution/standard_gaussian.hpp>

namespace fl
{

/**
 * \ingroup distribution_interfaces
 */
template <typename Variate, typename StandardVariate = internal::Empty>
class StandardGaussianMapping
        : public Sampling<Variate>
{
public:
    explicit StandardGaussianMapping(
            size_t snv_dimension = DimensionOf<StandardVariate>())
        : standard_gaussian_(snv_dimension)
    { }

    virtual ~StandardGaussianMapping() { }

    virtual Variate map_standard_normal(const StandardVariate& sample) const = 0;

    virtual Variate sample()
    {
        return map_standard_normal(standard_gaussian_.sample());
    }

    virtual int standard_variate_dimension() const
    {
        return standard_gaussian_.dimension();
    }

    virtual void standard_variate_dimension(size_t snv_dimension)
    {
        standard_gaussian_.dimension(snv_dimension);
    }

private:
    StandardGaussian<StandardVariate> standard_gaussian_;
};

// specialization for scalar noise
/**
 * \ingroup distribution_interfaces
 */
template <typename Variate>
class StandardGaussianMapping<Variate, double>:
        public Sampling<Variate>
{
public:
    virtual ~StandardGaussianMapping() { }

    virtual Variate map_standard_normal(const double& sample) const = 0;

    virtual Variate sample()
    {
        return map_standard_normal(standard_gaussian_.sample());
    }

    virtual int standard_variate_dimension() const
    {
        return 1;
    }
private:
    StandardGaussian<double> standard_gaussian_;
};



// specialization for no noise
/**
 * \ingroup distribution_interfaces
 */
template <typename Variate>
class StandardGaussianMapping<Variate, internal::Empty>:
        public Sampling<Variate>
{
public:
    virtual ~StandardGaussianMapping() { }

    virtual Variate map_standard_normal() const = 0;

    virtual Variate sample()
    {
        return map_standard_normal();
    }

    virtual int standard_variate_dimension() const
    {
        return 0;
    }
};


}

#endif
