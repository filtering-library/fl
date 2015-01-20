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
 * \file gaussian_map.hpp
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
template <typename Variate, typename NormalVariate = internal::Empty>
class GaussianMap:
        public Sampling<Variate>
{
public:
    explicit GaussianMap(const unsigned& noise_dimension = NormalVariate::SizeAtCompileTime):
        standard_gaussian_(noise_dimension)
    { }

    virtual ~GaussianMap() { }

    virtual Variate map_standard_normal(const NormalVariate& sample) const = 0;

    virtual Variate sample()
    {
        return map_standard_normal(standard_gaussian_.sample());
    }

    virtual int variate_dimension() const
    {
        return standard_gaussian_.dimension();
    }

    virtual void variate_dimension(size_t new_noise_dimension)
    {
        standard_gaussian_.dimension(new_noise_dimension);
    }

private:
    StandardGaussian<NormalVariate> standard_gaussian_;
};

// specialization for scalar noise
/**
 * \ingroup distribution_interfaces
 */
template <typename Variate>
class GaussianMap<Variate, double>:
        public Sampling<Variate>
{
public:
    virtual ~GaussianMap() { }

    virtual Variate map_standard_normal(const double& sample) const = 0;

    virtual Variate sample()
    {
        return map_standard_normal(standard_gaussian_.sample());
    }

    virtual int variate_dimension() const
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
class GaussianMap<Variate, internal::Empty>:
        public Sampling<Variate>
{
public:
    virtual ~GaussianMap() { }

    virtual Variate map_standard_normal() const = 0;

    virtual Variate sample()
    {
        return map_standard_normal();
    }

    virtual int variate_dimension() const
    {
        return 0;
    }
};


}

#endif
