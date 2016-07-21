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
template <typename ... Models> class JointTransition;

/**
 * Traits of JointTransition<MultipleOf<Transition, Count>>
 */
template <
    typename Transition,
    int Count
>
struct Traits<
           JointTransition<MultipleOf<Transition, Count>>
        >
{
    enum : signed int { ModelCount = Count };

    typedef Transition LocalTransition;
    typedef typename Traits<Transition>::Scalar Scalar;
    typedef typename Traits<Transition>::State LocalState;
    typedef typename Traits<Transition>::Input LocalInput;
    typedef typename Traits<Transition>::Noise LocalNoise;

    enum : signed int
    {
        StateDim = ExpandSizes<LocalState::SizeAtCompileTime, Count>::Size,
        NoiseDim = ExpandSizes<LocalNoise::SizeAtCompileTime, Count>::Size,
        InputDim = ExpandSizes<LocalInput::SizeAtCompileTime, Count>::Size
    };

    typedef Eigen::Matrix<Scalar, StateDim, 1> State;
    typedef Eigen::Matrix<Scalar, NoiseDim, 1> Noise;
    typedef Eigen::Matrix<Scalar, InputDim, 1> Input;

    typedef TransitionInterface<
                State,
                Noise,
                Input
            > TransitionBase;
};

/**
 * \ingroup process_models
 */
template <
    typename LocalTransition,
    int Count
>
class JointTransition<MultipleOf<LocalTransition, Count>>
    : public Traits<
                 JointTransition<MultipleOf<LocalTransition, Count>>
             >::TransitionBase
{
private:
    /** Typdef of \c This for #from_traits(TypeName) helper */
    typedef JointTransition This;

public:
    typedef from_traits(State);
    typedef from_traits(Noise);
    typedef from_traits(Input);

public:
    JointTransition(const LocalTransition& local_process_model,
                      int count = ToDimension<Count>())
        : local_process_model_(local_process_model),
          count_(count)
    {
        assert(count_ > 0);
    }

    JointTransition(const MultipleOf<LocalTransition, Count>& mof)
        : local_process_model_(mof.instance),
          count_(mof.count)
    {
        assert(count_ > 0);
    }

    virtual ~JointTransition() noexcept { }

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

    LocalTransition& local_process_model()
    {
        return local_process_model_;
    }

    const LocalTransition& local_process_model() const
    {
        return local_process_model_;
    }

protected:
    LocalTransition local_process_model_;
    int count_;
};

}



