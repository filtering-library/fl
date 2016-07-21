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
 * \file adaptive_model.hpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once


#include <fl/util/types.hpp>

namespace fl
{

/**
 * \ingroup transitions
 * \ingroup sensors
 * \interface AdaptiveModel
 *
 * \brief The AdaptiveModel interface represents the base of any model which
 * provides a variable set of parameters \f$\theta\f$ such that the model \f$y =
 * h(x, w)\f$ becomes \f$h(x, w, \theta)\f$. Here \f$x\f$ is the state,
 * \f$w\sim {\cal N}(0, 1)\f$ is a white noise term.
 *
 * Strictly speeking, the parameters \f$\theta\f$ are simply a part of the state
 * \f$x\f$. However, this separation is particulalry useful if the adaptation of
 * the paramters \f$\theta\f$ is optional. That is, \f$\theta\f$ may as well
 * remain unchanged during run-time without modifying the variable \f$x\f$ nor
 * forcing the extension of \f$x\f$ by the \f$\theta\f$.
 *
 * \note
 *
 * \tparam Parameter    Model variable parameters \f$\theta\f$. It represents a
 *                      subset of the models parameter which are subject to
 *                      changes during exectution time. That is, \f$\theta\f$
 *                      does not represent all model parameters but only those
 *                      which may be adapted on-line such as in adaptive
 *                      estiamtion in which the parameters are estimated along
 *                      the state \f$x\f$ \todo REF.
 */
template <typename Param>
class AdaptiveModel
    : public internal::AdaptiveModelType
{
public:
    typedef internal::AdaptiveModelType AdaptivityType;

public:
    /**
     * \return Model parameters \f$\theta\f$.
     */
    virtual const Param& param() const = 0;

    /**
     * Sets the adaptive parameters of the model.
     *
     * \param param     New model parameter values \f$\theta\f$
     */
    virtual void param(Param params) = 0;

    /**
     * \return Parameter variable dimension: \f$dim(\theta)\f$
     */
    virtual int param_dimension() const = 0;
};

}


