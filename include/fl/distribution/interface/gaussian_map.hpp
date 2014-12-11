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

#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>

#include <fl/util/traits.hpp>
#include <fl/distribution/interface/sampling.hpp>
#include <fl/distribution/standard_gaussian.hpp>

namespace fl
{


template <typename Vector, typename Noise = internal::Empty>
class GaussianMap:
        public Sampling<Vector>
{
public:
    explicit GaussianMap(const unsigned& noise_dimension = Noise::SizeAtCompileTime):
        standard_gaussian_(noise_dimension)
    { }

    virtual ~GaussianMap() { }

    virtual Vector MapStandardGaussian(const Noise& sample) const = 0;

    virtual Vector Sample()
    {
        return MapStandardGaussian(standard_gaussian_.Sample());
    }

    virtual int NoiseDimension() const
    {
        return standard_gaussian_.Dimension();
    }

    virtual void NoiseDimension(size_t new_noise_dimension)
    {
        standard_gaussian_.Dimension(new_noise_dimension);
    }

private:
    StandardGaussian<Noise> standard_gaussian_;
};

// specialization for scalar noise
template <typename Vector>
class GaussianMap<Vector, double>:
        public Sampling<Vector>
{
public:
    virtual ~GaussianMap() { }

    virtual Vector MapStandardGaussian(const double& sample) const = 0;

    virtual Vector Sample()
    {
        return MapStandardGaussian(standard_gaussian_.Sample());
    }

    virtual int NoiseDimension() const
    {
        return 1;
    }
private:
    StandardGaussian<double> standard_gaussian_;
};



// specialization for no noise
template <typename Vector>
class GaussianMap<Vector, internal::Empty>:
        public Sampling<Vector>
{
public:
    virtual ~GaussianMap() { }

    virtual Vector MapStandardGaussian() const = 0;

    virtual Vector Sample()
    {
        return MapStandardGaussian();
    }

    virtual int NoiseDimension() const
    {
        return 0;
    }
};


}

#endif
