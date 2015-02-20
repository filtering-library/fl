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
 *
 * FilterInterface represents the common interface of all filters. Each filter
 * must implement the two filter steps, predict/propagate and update.
 * The filtered state, observation and the state distribution are subject to the
 * underlying filtering algorithm and therefore may differ from one to another.
 * This interface, provides these types of the actual algorithm, such as the
 * the State, Observation, and StateDistribution types. Therefore, the traits
 * of each filter implememntation must provide the following types
 *
 * Required Types        | Description                        | Requirements
 * --------------------- | ---------------------------------- | ---------------
 * \c State              | Used State type                    | -
 * \c Input              | Process control input type         | -
 * \c Observation        | Used Observation type              | -
 * \c StateDistribution  | Distribution type over the state   | must implement fl::Moments
 * \c Ptr                | Shared pointer of the derived type | must specialize std::shared_ptr<>
 */
template <typename Derived>
class FilterInterface
{
public:
    /**
     * \brief Ptr is the shared pointer type of the filter specialization
     *
     * The filter specialization traits must provide the shared pointer type
     */
    typedef typename Traits<Derived>::Ptr Ptr;

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
     * \brief Observation type provided by the filter specialization
     *
     * The filter specialization traits must provide the \c Observation type.
     * The \c Observation type is commonly provided by the measurement
     * model.
     */
    typedef typename Traits<Derived>::Observation Observation;

    /**
     * \brief StateDistribution type uses by the filter specialization.
     *
     * The filter specialization uses a suitable state distribution. By
     * convension, the StateDistribution must implement the Moments
     * interface which provides the first two moments.
     */
    typedef typename Traits<Derived>::StateDistribution StateDistribution;

    /**
     * Predicts the distribution over the state give a delta time in seconds
     *
     * \param delta_time        Delta time of prediction
     * \param input             Control input argument
     * \param prior_dist        Prior state distribution     
     * \param predicted_dist    Predicted state distribution
     */
    virtual void predict(double delta_time,
                         const Input& input,
                         const StateDistribution& prior_dist,
                         StateDistribution& predicted_dist) = 0;

    /**
     * Updates a predicted state given an observation
     *
     * \param predicted_dist    Predicted state distribution
     * \param observation       Latest observation
     * \param posterior_dist    Updated posterior state distribution
     */
    virtual void update(const Observation& observation,
                        const StateDistribution& predicted_dist,
                        StateDistribution& posterior_dist) = 0;

    /**
     * Predicts the state distribution for a given delta time and subsequently
     * updates the prediction using a measurement.
     *
     * \param delta_time
     * \param input
     * \param observation
     * \param prior_dist
     * \param posterior_dist
     */
    virtual void predict_and_update(double delta_time,
                                    const Input& input,
                                    const Observation& observation,
                                    const StateDistribution& prior_dist,
                                    StateDistribution& posterior_dist) = 0;
};

}

#endif
