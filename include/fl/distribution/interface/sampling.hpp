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
 * \file sampling.hpp
 * \date May 2014
 * \author Jan Issac (jan.issac@gmail.com)
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#ifndef FL__DISTRIBUTION__INTERFACE__SAMPLING_HPP
#define FL__DISTRIBUTION__INTERFACE__SAMPLING_HPP

namespace fl
{

/**
 * \interface Sampling
 * \brief Distribution sampling interface
 *
 * \tparam Vector   Sample variable type
 */
template <typename Vector>
class Sampling
{
public:    
    /**
     * @return A random sample of the underlying distribution
     */
    virtual Vector Sample() = 0;

    /**
     * \brief Overridable default destructor
     */
    virtual ~Sampling() { }
};

}

#endif
