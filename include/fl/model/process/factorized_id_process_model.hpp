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
 * \file factorized_id_process_model.hpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__MODEL__PROCESS__FACTORIZED_ID_PROCESS_MODEL_HPP
#define FL__MODEL__PROCESS__FACTORIZED_ID_PROCESS_MODEL_HPP

#include <Eigen/Dense>

#include <tuple>
#include <memory>

#include <fl/util/traits.hpp>
#include <fl/util/meta.hpp>
#include <fl/distribution/gaussian.hpp>

#include <fl/model/process/process_model_interface.hpp>

namespace fl
{

// Forward declarations
template <typename...Models> class FactorizedIDProcessModel;

/**
 * Traits of FactorizedIDProcessModel
 */
template <typename...Models>
struct Traits<
           FactorizedIDProcessModel<Models...>
        >
{        
    enum : signed int
    {
        StateDim = JoinSizes<Traits<Models>::State::SizeAtCompileTime...>::Size,
        NoiseDim = JoinSizes<Traits<Models>::Noise::SizeAtCompileTime...>::Size,
        InputDim = JoinSizes<Traits<Models>::Input::SizeAtCompileTime...>::Size
    };

    typedef typename FirstTypeIn<Models...>::Type::State::Scalar Scalar;

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
template <typename ... Models>
class FactorizedIDProcessModel
    : public Traits<
                 FactorizedIDProcessModel<Models...>
             >::ProcessModelBase
{
public:
    typedef FactorizedIDProcessModel<Models...> This;

    typedef typename Traits<This>::State State;    
    typedef typename Traits<This>::Noise Noise;
    typedef typename Traits<This>::Input Input;

public:
    /**
     * Constructor a FactorizedIDProcessModel which is a composition of
     * mixture of fixed and dynamic-size process models. If
     *
     * \param models    Variadic list of shared pointers of the models
     */
    explicit
    FactorizedIDProcessModel(std::shared_ptr<Models>...models)
        : models_(models...)
    { }

    /**
     * \copydoc ProcessModelInterface::predict_state
     */
    virtual State predict_state(double delta_time,
                                const State& state,
                                const Noise& noise,
                                const Input& input)
    {
        State prediction = State(state_dimension(), 1);

        VariadicExpansion<sizeof...(Models)>().predict_state(
            models_,
            delta_time,
            state,
            noise,
            input,
            prediction);

        return prediction;
    }

    /**
     * \brief Overridable default destructor
     */
    ~FactorizedIDProcessModel() { }

    /**
     * \copydoc ProcessModelInterface::state_dimension
     *
     * Determines the joint size of the state vector of all models
     */
    virtual constexpr size_t state_dimension() const
    {
        return VariadicExpansion<sizeof...(Models)>().state_dimension(models_);
        //return expand_state_dimension(CreateIndexSequence<sizeof...(Models)>());
    }

    /**
     * \copydoc ProcessModelInterface::noise_dimension
     *
     * Determines the joint size of the noise vector of all models
     */
    virtual constexpr size_t noise_dimension() const
    {
        return VariadicExpansion<sizeof...(Models)>().noise_dimension(models_);
        //return expand_noise_dimension(CreateIndexSequence<sizeof...(Models)>());
    }

    /**
     * \copydoc ProcessModelInterface::input_dimension
     *
     * Determines the joint size of the input vector of all models
     */
    virtual constexpr size_t input_dimension() const
    {
        return VariadicExpansion<sizeof...(Models)>().input_dimension(models_);
        //return expand_input_dimension(CreateIndexSequence<sizeof...(Models)>());
    }

protected:
    /**
     * \brief Contains the points to the sub-models which this joint model
     *        is composed of.
     */
    std::tuple<std::shared_ptr<Models>...> models_;


private:
    /** \cond INTERNAL */

//    template <int...Indices>
//    constexpr size_t expand_state_dimension(IndexSequence<Indices...>) const
//    {
//        int joint_dim = 0;

//        auto&& dims = { std::get<Indices>(models_)->state_dimension()... };

//        for (auto dim : dims) { joint_dim += dim; }

//        return joint_dim;
//    }

//    template <int...Indices>
//    constexpr size_t expand_noise_dimension(IndexSequence<Indices...>) const
//    {
//        int joint_dim = 0;

//        auto&& dims = { std::get<Indices>(models_)->noise_dimension()... };

//        for (auto dim : dims) { joint_dim += dim; }

//        return joint_dim;
//    }

//    template <int...Indices>
//    constexpr size_t expand_input_dimension(IndexSequence<Indices...>) const
//    {
//        int joint_dim = 0;

//        auto&& dims = { std::get<Indices>(models_)->input_dimension()... };

//        for (auto dim : dims) { joint_dim += dim; }

//        return joint_dim;
//    }

    /**
     * Represents expansion of model function calls on the variadic model list
     */
    template <int k, bool make_non_explicit = true>
    struct VariadicExpansion
    {
        template <typename Tuple>
        constexpr size_t state_dimension(Tuple& models_tuple) const
        {
            return ::fl::join_sizes(
                        std::get<k - 1>(models_tuple)->state_dimension(),
                        VariadicExpansion<k - 1, make_non_explicit>()
                            .state_dimension(models_tuple));
        }

        template <typename Tuple>
        constexpr size_t noise_dimension(Tuple& models_tuple) const
        {
            return ::fl::join_sizes(
                        std::get<k - 1>(models_tuple)->noise_dimension(),
                        VariadicExpansion<k - 1, make_non_explicit>()
                            .noise_dimension(models_tuple));
        }

        template <typename Tuple>
        constexpr size_t input_dimension(Tuple& models_tuple) const
        {
            return ::fl::join_sizes(
                        std::get<k - 1>(models_tuple)->input_dimension(),
                        VariadicExpansion<k - 1, make_non_explicit>()
                            .input_dimension(models_tuple));
        }

        template <
            typename Tuple,
            typename State,
            typename Noise,
            typename Input
        >
        void predict_state(Tuple& models_tuple,
                           double delta_time,
                           const State& state,
                           const Noise& noise,
                           const Input& input,
                           State& prediction,
                           int state_offset = 0,
                           int noise_offset = 0,
                           int input_offset = 0)
        {
            auto&& model = std::get<k - 1>(models_tuple);

            auto state_dim = model->state_dimension();
            auto noise_dim = model->noise_dimension();
            auto input_dim = model->input_dimension();

            state_offset += state_dim;
            noise_offset += noise_dim;
            input_offset += input_dim;

            prediction.middleRows(state.rows() - state_offset, state_dim) =
                model->predict_state(
                    delta_time,
                    state.middleRows(state.rows() - state_offset, state_dim),
                    noise.middleRows(noise.rows() - noise_offset, noise_dim),
                    input.middleRows(input.rows() - input_offset, input_dim)
                );

            VariadicExpansion<k - 1, make_non_explicit>()
                .predict_state(
                    models_tuple,
                    delta_time,
                    state,
                    noise,
                    input,
                    prediction,
                    state_offset,
                    noise_offset,
                    input_offset
                );
        }
    };

    /**
     * Terminal expansion specialization.
     *
     * Uses void implementation to avoid redundancy
     */
    template <bool make_non_explicit>
    struct VariadicExpansion<0, make_non_explicit>
    {
        template <typename Tuple>
        constexpr size_t state_dimension(Tuple&) const { return 0; }

        template <typename Tuple>
        constexpr size_t noise_dimension(Tuple&) const { return 0; }

        template <typename Tuple>
        constexpr size_t input_dimension(Tuple&) const { return 0; }

        template <
            typename Tuple,
            typename State,
            typename Noise,
            typename Input
        >
        void predict_state(Tuple& models_tuple,
                           double delta_time,
                           const State& state,
                           const Noise& noise,
                           const Input& input,
                           State& prediction,
                           int state_offset,
                           int noise_offset,
                           int input_offset)
        { }
    };
    /** \endcond */
};

}

#endif

