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
 * \file factorized_iid_process_model.hpp
 * \date January 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__MODEL__PROCESS__FACTORIZED_IID_PROCESS_MODEL_HPP
#define FL__MODEL__PROCESS__FACTORIZED_IID_PROCESS_MODEL_HPP

#include <Eigen/Dense>

#include <memory>

#include <fl/util/traits.hpp>
#include <fl/util/meta.hpp>
#include <fl/distribution/gaussian.hpp>

#include <fl/model/process/process_model_interface.hpp>

namespace fl
{

// Forward declarations
template <    
    typename ProcessModel,
    int Factors
>
class FactorizedIIDProcessModel;

/**
 * Traits of FactorizedIIDProcessModel
 */
template <    
    typename ProcessModel,
    int Factors
>
struct Traits<
           FactorizedIIDProcessModel<ProcessModel, Factors>
        >
{
    static constexpr int IIDFactors = Factors;

    typedef ProcessModel LocalProcessModel;
    typedef typename Traits<ProcessModel>::Scalar Scalar;
    typedef typename Traits<ProcessModel>::State LocalState;
    typedef typename Traits<ProcessModel>::Input LocalInput;
    typedef typename Traits<ProcessModel>::Noise LocalNoise;

    typedef Eigen::Matrix<
                Scalar,
                FactorSize<LocalState::RowsAtCompileTime, Factors>::Size,
                1
            > State;

    typedef Eigen::Matrix<
                Scalar,
                FactorSize<LocalNoise::RowsAtCompileTime, Factors>::Size,
                1
            > Noise;

    typedef Eigen::Matrix<
                Scalar,
                FactorSize<LocalInput::RowsAtCompileTime, Factors>::Size,
                1
            > Input;

    typedef ProcessModelInterface<
                State,
                Noise,
                Input
            > ProcessModelBase;
};

/**
 * \ingroup observation_models
 */
template <
    typename LocalProcessModel,
    int Factors = Eigen::Dynamic
>
class FactorizedIIDProcessModel
    : public Traits<
                 FactorizedIIDProcessModel<LocalProcessModel, Factors>
             >::ProcessModelBase
{
public:
    typedef FactorizedIIDProcessModel<LocalProcessModel, Factors> This;

    typedef typename Traits<This>::State State;    
    typedef typename Traits<This>::Noise Noise;
typedef typename Traits<This>::Input Input;

public:
    FactorizedIIDProcessModel(
            const std::shared_ptr<LocalProcessModel>& local_process_model,
            int factors = Factors)
        : local_process_model_(local_process_model),
          factors_(factors)
    {
        assert(factors_ > 0);
    }

    ~FactorizedIIDProcessModel() { }

    virtual State predict_state(double delta_time,
                                const State& state,
                                const Noise& noise,
                                const Input& input)
    {
        State x = State::Zero(state_dimension(), 1);

        int noise_dim = local_process_model_->noise_dimension();
        int state_dim = local_process_model_->state_dimension();
        int input_dim = local_process_model_->input_dimension();

        for (int i = 0; i < factors_; ++i)
        {
            x.middleRows(i * state_dim, state_dim) =
                local_process_model_->predict_state(
                    delta_time,
                    state.middleRows(i * state_dim, state_dim),
                    noise.middleRows(i * noise_dim, noise_dim),
                    input.middleRows(i * input_dim, input_dim));
        }

        return x;
    }    

    virtual size_t state_dimension() const
    {
        return local_process_model_->state_dimension() * factors_;
    }

    virtual size_t noise_dimension() const
    {
        return local_process_model_->noise_dimension() * factors_;
    }

    virtual size_t input_dimension() const
    {
        return local_process_model_->input_dimension() * factors_;
    }

    const std::shared_ptr<LocalProcessModel>& local_process_model()
    {
        return local_process_model_;
    }

protected:
    std::shared_ptr<LocalProcessModel> local_process_model_;
    size_t factors_;
};

}

#endif

