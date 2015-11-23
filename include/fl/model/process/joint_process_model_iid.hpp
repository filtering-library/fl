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
 * \file joint_process_model_iid.hpp
 * \date January 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once


#include <Eigen/Dense>

#include <utility>

#include <fl/util/traits.hpp>
#include <fl/util/meta.hpp>

#include <fl/model/process/interface/state_transition_function.hpp>

namespace fl
{

// Forward declarations
template <typename ... Models> class JointProcessModel;

/**
 * Traits of JointProcessModel<MultipleOf<ProcessModel, Count>>
 */
template <
    typename ProcessModel,
    int Count
>
struct Traits<
           JointProcessModel<MultipleOf<ProcessModel, Count>>
        >
{
    enum : signed int { ModelCount = Count };

    typedef ProcessModel LocalProcessModel;
    typedef typename Traits<ProcessModel>::Scalar Scalar;
    typedef typename Traits<ProcessModel>::State LocalState;
    typedef typename Traits<ProcessModel>::Input LocalInput;
    typedef typename Traits<ProcessModel>::Noise LocalNoise;

    enum : signed int
    {
        StateDim = ExpandSizes<LocalState::SizeAtCompileTime, Count>::Size,
        NoiseDim = ExpandSizes<LocalNoise::SizeAtCompileTime, Count>::Size,
        InputDim = ExpandSizes<LocalInput::SizeAtCompileTime, Count>::Size
    };

    typedef Eigen::Matrix<Scalar, StateDim, 1> State;
    typedef Eigen::Matrix<Scalar, NoiseDim, 1> Noise;
    typedef Eigen::Matrix<Scalar, InputDim, 1> Input;

    typedef ProcessModelInterface<
                State,
                Noise,
                Input
            > ProcessModelBase;
};

/**
 * \ingroup process_models
 */
template <
    typename LocalProcessModel,
    int Count
>
class JointProcessModel<MultipleOf<LocalProcessModel, Count>>
    : public Traits<
                 JointProcessModel<MultipleOf<LocalProcessModel, Count>>
             >::ProcessModelBase
{
private:
    /** Typdef of \c This for #from_traits(TypeName) helper */
    typedef JointProcessModel This;

public:
    typedef from_traits(State);
    typedef from_traits(Noise);
    typedef from_traits(Input);

public:
    JointProcessModel(const LocalProcessModel& local_process_model,
                      int count = ToDimension<Count>())
        : local_process_model_(local_process_model),
          count_(count)
    {
        assert(count_ > 0);
    }

    JointProcessModel(const MultipleOf<LocalProcessModel, Count>& mof)
        : local_process_model_(mof.instance),
          count_(mof.count)
    {
        assert(count_ > 0);
    }

    virtual ~JointProcessModel() noexcept { }

    virtual State predict_state(double delta_time,
                                const State& state,
                                const Noise& noise,
                                const Input& input)
    {
        State x = State::Zero(state_dimension(), 1);

        int state_dim = local_process_model_.state_dimension();
        int noise_dim = local_process_model_.noise_dimension();
        int input_dim = local_process_model_.input_dimension();

        for (int i = 0; i < count_; ++i)
        {
            x.middleRows(i * state_dim, state_dim) =
                local_process_model_.predict_state(
                    delta_time,
                    state.middleRows(i * state_dim, state_dim),
                    noise.middleRows(i * noise_dim, noise_dim),
                    input.middleRows(i * input_dim, input_dim));
        }

        return x;
    }

    virtual int state_dimension() const
    {
        return local_process_model_.state_dimension() * count_;
    }

    virtual int noise_dimension() const
    {
        return local_process_model_.noise_dimension() * count_;
    }

    virtual int input_dimension() const
    {
        return local_process_model_.input_dimension() * count_;
    }

    LocalProcessModel& local_process_model()
    {
        return local_process_model_;
    }

    const LocalProcessModel& local_process_model() const
    {
        return local_process_model_;
    }

protected:
    LocalProcessModel local_process_model_;
    int count_;
};

}



