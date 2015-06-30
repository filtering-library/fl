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

#ifndef FL__DISTRIBUTION__INTERFACE__STANDARD_GAUSSIAN_MAPPING_HPP
#define FL__DISTRIBUTION__INTERFACE__STANDARD_GAUSSIAN_MAPPING_HPP

#include <Eigen/Dense>

#include <type_traits>

#include <fl/util/traits.hpp>
#include <fl/distribution/interface/sampling.hpp>
#include <fl/distribution/standard_gaussian.hpp>

namespace fl
{

/**
 * \ingroup distribution_interfaces
 *
 * \brief Represents the interface which provides a mapping
 *        from a standard normal variate onto the underlying distribution which
 *        implements this interface
 *
 * \tparam Variate              The Distribution variate type. This is the type
 *                              which is being returned by the mapping or
 *                              sampling, respectively.
 * \tparam StdVariateDimension  Dimension of the source variate type which is
 *                              the standard normal variate
 *                              \f$x_{SNV}\sim{\cal N}(0, I)\f$
 */
template <typename Variate, int StdVariateDimension>
class StandardGaussianMapping
    : public Sampling<Variate>
{
public:
    typedef Eigen::Matrix<Real, StdVariateDimension, 1> StandardVariate;

    /**
     * StandardGaussianMapping constructor. It initializes the mapper
     *
     * \param snv_dimension     Dimension of the standard normal variate
     */
    explicit StandardGaussianMapping(
            int snv_dimension = DimensionOf<StandardVariate>())
        : standard_gaussian_(snv_dimension)
    { }

    /**
     * \brief Overridable default destructor
     */
    virtual ~StandardGaussianMapping() { }

    /**
     * \brief Mapps a standard normal variate onto a sample of the underlying
     *        distribution which implements this mapper
     *
     * \param sample  SNV sample which will be mapped onto a variate sampe
     *
     * \return A variate according to the underlying distribution
     */
    virtual Variate map_standard_normal(const StandardVariate& sample) const = 0;

    /**
     * \return A variate according to the underlying distribution
     */
    virtual Variate sample() const
    {
        return map_standard_normal(standard_gaussian_.sample());
    }

    /**
     * \return Dimension of the standard normal variate used for mapping
     */
    virtual int standard_variate_dimension() const
    {
        return standard_gaussian_.dimension();
    }

    /**
     * \brief Sets the dimension of the standard normal variate
     *
     * \param snv_dimension The new dimension of the SNV
     */
    virtual void standard_variate_dimension(int snv_dimension)
    {
        standard_gaussian_.dimension(snv_dimension);
    }

protected:
    /**
     * \brief SNV generator
     */
    mutable StandardGaussian<StandardVariate> standard_gaussian_;
};

/**
 * \ingroup distribution_interfaces
 *
 * \brief Represents the interface which provides a mapping
 *        from a scalar standard normal variate onto the underlying distribution
 *        which implements this interface.
 *
 * \tparam Variate          The Distribution variate type. This is the type
 *                          which is being returned by the mapping or sampling,
 *                          respectively.
 */
template <typename Variate>
class StandardGaussianMapping<Variate, 1>
    : public Sampling<Variate>
{
public:
    typedef Real StandardVariate;

    /**
     * \brief Overridable default destructor
     */
    virtual ~StandardGaussianMapping() { }

    /**
     * \brief Mapps a one dimensional standard normal variate onto a sample of
     *        the underlying distribution which implements this mapper
     *
     * \param sample SNV sample which will be mapped onto a variate sampe
     *
     * \return A variate according to the underlying distribution
     */
    virtual Variate map_standard_normal(const Real& sample) const = 0;

    /**
     * \return A variate according to the underlying distribution
     */
    virtual Variate sample() const
    {
        return map_standard_normal(standard_gaussian_.sample());
    }

    /**
     * \return Dimension of the standard normal variate used for mapping
     */
    virtual int standard_variate_dimension() const
    {
        return 1;
    }

protected:
    /**
     * \brief One dimensional SNV generator
     */
    mutable StandardGaussian<Real> standard_gaussian_;
};

}

#endif
