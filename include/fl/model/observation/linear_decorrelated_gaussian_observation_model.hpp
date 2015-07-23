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
 * \file linear_decorrelated_gaussian_observation_model.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__MODEL__OBSERVATION__LINEAR_DECORRELATED_GAUSSIAN_OBSERVATION_MODEL_HPP
#define FL__MODEL__OBSERVATION__LINEAR_DECORRELATED_GAUSSIAN_OBSERVATION_MODEL_HPP

#include <fl/util/traits.hpp>
#include <fl/util/descriptor.hpp>
#include <fl/distribution/decorrelated_gaussian.hpp>
#include <fl/model/observation/linear_observation_model.hpp>
#include <fl/model/observation/interface/additive_uncorrelated_observation_function.hpp>

namespace fl
{

/**
 * \ingroup observation_models
 */
template <typename Obsrv, typename State>
class LinearDecorrelatedGaussianObservationModel<Obsrv, State>
    : public LinearObservationModel<Obsrv, State, DecorrelatedGaussian<Obsrv>>,
      public AdditiveDecorrelatedObservationFunction<Obsrv, State, Obsrv>,
      public Descriptor
{
public:

    typedef AdditiveDecorrelatedObservationFunction<
                Obsrv, State, Obsrv
            > AdditiveDecorrelatedInterface;

    typedef typename AdditiveDecorrelatedInterface::NoiseDiagonal NoiseDiagonal;

    using AdditiveDecorrelatedInterface::covariance;
    using AdditiveDecorrelatedInterface::square_root;

    /**
     * Constructs a linear gaussian observation model
     *
     * \param obsrv_dim     observation dimension if dynamic size
     * \param state_dim     state dimension if dynamic size
     */
    explicit
    LinearDecorrelatedGaussianObservationModel(
        int obsrv_dim = DimensionOf<Obsrv>(),
        int state_dim = DimensionOf<State>())
        : LinearObservationModel<Obsrv, State, DecorrelatedGaussian<Obsrv>>(
              obsrv_dim, state_dim)
    { }

    virtual const NoiseDiagonal& noise_matrix_diagonal() const
    {
        return AdditiveDecorrelatedInterface::square_root();
    }

    virtual const NoiseDiagonal& noise_covariance_diagonal() const
    {
        return AdditiveDecorrelatedInterface::covariance();
    }

    virtual std::string name() const
    {
        return "LinearDecorrelatedGaussianObservationModel";
    }

    virtual std::string description() const
    {
        return "Linear observation model with additive decorrelated Gaussian "
               "noise";
    }
};

}

#endif
