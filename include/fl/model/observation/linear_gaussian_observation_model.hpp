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
 * \file linear_gaussian_observation_model.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__MODEL__OBSERVATION__LINEAR_GAUSSIAN_OBSERVATION_MODEL_HPP
#define FL__MODEL__OBSERVATION__LINEAR_GAUSSIAN_OBSERVATION_MODEL_HPP

#include <fl/util/traits.hpp>
#include <fl/util/descriptor.hpp>
#include <fl/distribution/gaussian.hpp>
#include <fl/model/observation/linear_observation_model.hpp>

namespace fl
{

/**
 * \ingroup observation_models
 *
 * This represents the linear gaussian observation model \f$p(y_t \mid x_t)\f$
 * governed by the linear equation
 *
 * \f$ y_t  = H_t x_t + N_t v_t \f$
 *
 * where \f$ y_t\f$ is the observation, \f$ x_t\f$ is the state. The matrix
 * \f$H_t\f$ is the sensor model mapping the state into the observation space.
 * The vector \f$ v_t \sim {\cal N}(v_t; 0, I)\f$ is a standard normal variate
 * which is mapped into the the observation space via the noise model matrix
 * \f$N_t\f$. Any Gaussian noise \f$\tilde{v}_t \sim {\cal N}(\tilde{v}_t ; 0,
 * R_t) \f$ can be represented via \f$\tilde{v}_t = N_t v_t\f$ with\f$N_t
 * = \sqrt{R_t}\f$. Hence, the linear equation may be restated as the more
 * familiar but equivalent form
 *
 * \f$ y_t  = H_t x_t + \tilde{v}_t \f$.
 */
template <typename Obsrv, typename State>
class LinearGaussianObservationModel
    : public LinearObservationModel<Obsrv, State, Gaussian<Obsrv>>,
      public Descriptor
{
public:
    /**
     * Constructs a linear gaussian observation model
     * \param obsrv_dim     observation dimension if dynamic size
     * \param state_dim     state dimension if dynamic size
     */
    explicit
    LinearGaussianObservationModel(int obsrv_dim = DimensionOf<Obsrv>(),
                                   int state_dim = DimensionOf<State>())
        : LinearObservationModel<Obsrv, State, Gaussian<Obsrv>>(
              obsrv_dim, state_dim)
    { }

    virtual std::string name() const
    {
        return "LinearGaussianObservationModel";
    }

    virtual std::string description() const
    {
        return "Linear observation model with additive Gaussian noise";
    }
};

}

#endif
