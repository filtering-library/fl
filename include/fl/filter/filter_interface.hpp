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
 * \file filter_interface.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__FILTER__FILTER_INTERFACE_HPP
#define FL__FILTER__FILTER_INTERFACE_HPP

#include <fl/util/traits.hpp>

namespace fl
{

/**
 * \interface FilterInterface
 *
 * \brief Common filter interface
 *
 *
 * FilterInterface represents the common interface of all filters. Each filter
 * must implement the two filter steps, predict/propagate and update.
 * The filtered state, observation and the state distribution are subject to the
 * underlying filtering algorithm and therefore may differ from one to another.
 * This interface, provides these types of the actual algorithm, such as the
 * the State, Obsrv (Observation), and Belief types. Therefore, the
 * traits of each filter implememntation must provide the following types
 *
 * Required Types | Description                        | Requirements
 * -------------- | ---------------------------------- | ---------------
 * \c State       | Used State type                    | -
 * \c Input       | Process control input type         | -
 * \c Obsrv       | Used observation type              | -
 * \c Belief      | Distribution type over the state   | implements fl::Moments
 */
template <typename Derived>
class FilterInterface
{
public:
    /**
     * \brief State type provided by the filter specialization
     *
     * The filter specialization traits must provide the \c State type. The
     * state type is commonly either provided by the process model or the filter
     * self.
     */
    typedef typename Traits<Derived>::State State;

    /**
     * \brief Input control type provided by the filter specialization
     *
     * The filter specialization traits must provide the \c Input control type.
     * The \c Input type is commonly either provided by the process model or the
     * filter self.
     */
    typedef typename Traits<Derived>::Input Input;

    /**
     * \brief Obsrv (Observation) type provided by the filter specialization
     *
     * The filter specialization traits must provide the \c Obsrv type.
     * The \c Obsrv type is commonly provided by the measurement
     * model.
     */
    typedef typename Traits<Derived>::Obsrv Obsrv;

    /**
     * \brief Belief type uses by the filter specialization.
     *
     * The filter specialization uses a suitable state distribution. By
     * convension, the Belief must implement the Moments
     * interface which provides the first two moments.
     */
    typedef typename Traits<Derived>::Belief Belief;

    /**
     * Predicts the distribution over the state for the next time step
     *
     * \param prior_belief        Prior state distribution
     * \param input               Control input argument
     * \param predicted_belief    Predicted state distribution
     */
    virtual void predict(const Belief& prior_belief,
                         const Input& input,
                         Belief& predicted_belief) = 0;

    /**
     * Predicts the distribution over the state give a number of time steps.
     * This may provide a more efficient implementation than simply calling
     * the \c predict function recursively.
     *
     * \param prior_belief        Prior state distribution
     * \param input               Control input argument
     * \param steps
     * \param predicted_belief    Predicted state distribution
     */
    virtual void predict(const Belief& prior_belief,
                         const Input& input,
                         const long steps,
                         Belief& predicted_belief)
    {
        predicted_belief = prior_belief;

        for (int i = 0; i < steps; ++i)
        {
            predict(predicted_belief, input, predicted_belief);
        }
    }

    /**
     * Updates a predicted state given an observation
     *
     * \param predicted_belief    Predicted state distribution
     * \param observation       Latest observation
     * \param posterior_belief    Updated posterior state distribution
     */
    virtual void update(const Belief& predicted_belief,
                        const Obsrv& obsrv,
                        Belief& posterior_belief) = 0;

    /**
     * Predicts the state distribution for a given delta time and subsequently
     * updates the prediction using a measurement.
     *
     * @param delta_time
     * @param input
     * @param observation
     * @param prior_belief
     * @param posterior_belief
     */
    virtual void predict_and_update(const Belief& prior_belief,
                                    const Input& input,
                                    const Obsrv& obsrv,
                                    Belief& posterior_belief) = 0;
};

}

#endif
