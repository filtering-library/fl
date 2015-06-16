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
 * \file additive_observation_function.hpp
 * \date June 2015
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__MODEL__OBSERVATION__ADDITIVE_OBSERVATION_FUNCTION_HPP
#define FL__MODEL__OBSERVATION__ADDITIVE_OBSERVATION_FUNCTION_HPP

#include <fl/util/traits.hpp>

#include <fl/model/observation/interface/observation_model_interface.hpp>

namespace fl
{

template <
    typename Obsrv,
    typename State,
    typename Noise,
    int Id = 0
>
class AdditiveObservationFunction
    : public ObservationFunction<Obsrv, State, Noise, Id>
{
public:

    /**
     * Noise model matrix \f$N_t\f$
     */
    typedef Eigen::Matrix<
            typename Noise::Scalar,
            Noise::SizeAtCompileTime,
            Noise::SizeAtCompileTime
        > NoiseMatrix;

    /**
     * Evaluates the model function \f$y = h(x, w)\f$ where \f$x\f$ is the state
     * and \f$w\sim {\cal N}(0, 1)\f$ is a white noise parameter. Put
     * differently, \f$y = h(x, w)\f$ is a sample from the conditional model
     * distribution \f$p(y \mid x)\f$.
     *
     * \param state         The state variable \f$x\f$
     * \param noise         The noise term \f$w\f$
     * \param delta_time    Prediction time
     */
    virtual Obsrv expected_observation(const State& state) const = 0;

    /**
     * \brief noise_model
     *
     * \return
     */
    virtual const NoiseMatrix& noise_matrix() const = 0;

    virtual Obsrv observation(const State& state,
                              const Noise& noise) const
    {
        return expected_observation(state) + noise_matrix() * noise;
    }
};

}

#endif
