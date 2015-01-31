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
 * \file factorized_iid_observation_model.hpp
 * \date January 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__MODEL__OBSERVATION__FACTORIZED_IID_OBSERVATION_MODEL_HPP
#define FL__MODEL__OBSERVATION__FACTORIZED_IID_OBSERVATION_MODEL_HPP

#include <Eigen/Dense>

#include <memory>

#include <fl/util/traits.hpp>
#include <fl/util/meta.hpp>
#include <fl/distribution/gaussian.hpp>

#include <fl/model/observation/observation_model_interface.hpp>

namespace fl
{

// Forward declarations
template <    
    typename ObservationModel,
    int Factors
>
class FactorizedIIDObservationModel;

/**
 * Traits of FactorizedIIDObservationModel
 */
template <    
    typename ObservationModel,
    int Factors
>
struct Traits<
           FactorizedIIDObservationModel<ObservationModel, Factors>
        >
{
    static constexpr int IIDFactors = Factors;

    typedef ObservationModel LocalObservationModel;
    typedef typename Traits<ObservationModel>::Scalar Scalar;
    typedef typename Traits<ObservationModel>::State LocalState;
    typedef typename Traits<ObservationModel>::Observation LocalObservation;

    typedef Eigen::Matrix<
                Scalar,
                FactorSize<LocalObservation::RowsAtCompileTime, Factors>::Size,
                1
            > Observation;

    typedef Eigen::Matrix<
                Scalar,
                FactorSize<LocalObservation::RowsAtCompileTime, Factors>::Size,
                1
            > Noise;

    typedef Eigen::Matrix<
                Scalar,
                FactorSize<LocalState::RowsAtCompileTime, Factors>::Size,
                1
            > State;

    typedef ObservationModelInterface<
                Observation,
                State,
                Noise
            > ObservationModelBase;
};

/**
 * \ingroup observation_models
 */
template <
    typename LocalObservationModel,
    int Factors
>
class FactorizedIIDObservationModel
    : public Traits<
                 FactorizedIIDObservationModel<LocalObservationModel, Factors>
             >::ObservationModelBase
{
public:
    typedef FactorizedIIDObservationModel<LocalObservationModel, Factors> This;

    typedef typename Traits<This>::State State;
    typedef typename Traits<This>::Observation Observation;
    typedef typename Traits<This>::Noise Noise;    

public:
    FactorizedIIDObservationModel(
            const std::shared_ptr<LocalObservationModel>& local_obsrv_model,
            int factors = Factors)
        : local_obsrv_model_(local_obsrv_model),
          factors_(factors)
    {
        auto H = local_obsrv_model_->H();
        H.setIdentity();
        local_obsrv_model_->H(H);
    }

    ~FactorizedIIDObservationModel() { }

    virtual Observation predict_observation(const State& state,
                                            const Noise& noise,
                                            double delta_time)
    {
        Observation y = Observation::Zero(observation_dimension(), 1);

        int obsrv_dim = local_obsrv_model_->observation_dimension();
        int noise_dim = local_obsrv_model_->noise_dimension();
        int state_dim = local_obsrv_model_->state_dimension();

        for (int i = 0; i < factors_; ++i)
        {
            y.middleRows(i * obsrv_dim, obsrv_dim) =
                local_obsrv_model_->predict_observation(
                    state.middleRows(i * state_dim, state_dim),
                    noise.middleRows(i * noise_dim, noise_dim),
                    delta_time);
        }

        return y;
    }

    virtual size_t observation_dimension() const
    {
        return local_obsrv_model_->observation_dimension() * factors_;
    }

    virtual size_t state_dimension() const
    {
        return local_obsrv_model_->state_dimension() * factors_;
    }

    virtual size_t noise_dimension() const
    {
        return local_obsrv_model_->noise_dimension() * factors_;
    }

    const std::shared_ptr<LocalObservationModel>& local_observation_model()
    {
        return local_obsrv_model_;
    }

protected:
    std::shared_ptr<LocalObservationModel> local_obsrv_model_;
    size_t factors_;
};

}

#endif

