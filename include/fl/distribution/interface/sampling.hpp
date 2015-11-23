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
 * \file sampling.hpp
 * \date May 2014
 * \author Jan Issac (jan.issac@gmail.com)
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#pragma once


namespace fl
{

/**
 * \ingroup distribution_interfaces
 *
 * \brief Distribution sampling interface
 *
 * \tparam Variate  Variate type of the random variable
 */
template <typename Variate>
class Sampling
{
public:
    /**
     * \brief Overridable default destructor
     */
    virtual ~Sampling() noexcept { }

    /**
     * \return A random sample of the underlying distribution \f[x \sim p(x)\f]
     */
    virtual Variate sample() const = 0;
};

}
