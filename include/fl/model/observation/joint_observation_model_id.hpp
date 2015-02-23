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
 * \file joint_observation_model_id.hpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__MODEL__OBSERVATION__JOINT_OBSERVATION_MODEL_ID_HPP
#define FL__MODEL__OBSERVATION__JOINT_OBSERVATION_MODEL_ID_HPP

#include <Eigen/Dense>

#include <tuple>
#include <memory>

#include <fl/util/traits.hpp>
#include <fl/util/meta.hpp>

#include <fl/model/observation/observation_model_interface.hpp>

namespace fl
{

// Forward declarations
template <typename...Models> class JointObservationModel;

/**
 * Traits of JointObservationModel
 */
template <typename...Models>
struct Traits<
           JointObservationModel<Models...>
       >
{
    enum : signed int
    {
        ObsrvDim = JoinSizes<Traits<Models>::Observation::SizeAtCompileTime...>::Size,
        StateDim = JoinSizes<Traits<Models>::State::SizeAtCompileTime...>::Size,
        NoiseDim = JoinSizes<Traits<Models>::Noise::SizeAtCompileTime...>::Size
    };

    typedef typename Traits<
                         typename FirstTypeIn<Models...>::Type
                     >::State::Scalar Scalar;

    typedef Eigen::Matrix<Scalar, StateDim, 1> State;
    typedef Eigen::Matrix<Scalar, NoiseDim, 1> Noise;
    typedef Eigen::Matrix<Scalar, ObsrvDim, 1> Observation;

    typedef ObservationModelInterface<
                Observation,
                State,
                Noise
            > ObservationModelBase;
};

/**
 * \ingroup process_models
 */
template <typename ... Models>
class JointObservationModel
    : public Traits<
                 JointObservationModel<Models...>
             >::ObservationModelBase
{
public:
    typedef JointObservationModel<Models...> This;

    typedef typename Traits<This>::State State;
    typedef typename Traits<This>::Noise Noise;
    typedef typename Traits<This>::Observation Observation;

public:
    /**
     * Constructor a JointObservationModel which is a composition of mixture of
     * fixed and dynamic-size observation models.
     *
     * \param models    Variadic list of shared pointers of the models
     */
    explicit
    JointObservationModel(std::shared_ptr<Models>...models)
        : models_(models...)
    { }

    /**
     * \brief Overridable default destructor
     */
    ~JointObservationModel() { }

    /**
     * \copydoc ObservationModelInterface::predict_observation
     */
    virtual Observation predict_observation(const State& state,
                                            const Noise& noise,
                                            double delta_time)
    {
        Observation prediction = Observation(observation_dimension(), 1);

        predict_observation<sizeof...(Models)>(
            models_,
            delta_time,
            state,
            noise,
            prediction);

        return prediction;
    }


    /**
     * \copydoc ObservationModelInterface::state_dimension
     *
     * Determines the joint size of the state vector of all models
     */
    virtual constexpr int state_dimension() const
    {
        return expand_state_dimension(CreateIndexSequence<sizeof...(Models)>());
    }

    /**
     * \copydoc ObservationModelInterface::noise_dimension
     *
     * Determines the joint size of the noise vector of all models
     */
    virtual constexpr int noise_dimension() const
    {
        return expand_noise_dimension(CreateIndexSequence<sizeof...(Models)>());
    }

    /**
     * \copydoc ObservationModelInterface::input_dimension
     *
     * Determines the joint size of the observation vector of all models
     */
    virtual constexpr int observation_dimension() const
    {
        return expand_obsrv_dimension(CreateIndexSequence<sizeof...(Models)>());
    }

protected:
    /**
     * \brief Contains the points to the sub-models which this joint model
     *        is composed of.
     */
    std::tuple<std::shared_ptr<Models>...> models_;


private:
    /** \cond INTERNAL */
    template <int...Indices>
    constexpr int expand_state_dimension(IndexSequence<Indices...>) const
    {
        const auto& dims = { std::get<Indices>(models_)->state_dimension()... };

        int joint_dim = 0;
        for (auto dim : dims) { joint_dim += dim; }

        return joint_dim;
    }

    template <int...Indices>
    constexpr int expand_noise_dimension(IndexSequence<Indices...>) const
    {
        const auto& dims = { std::get<Indices>(models_)->noise_dimension()... };

        int joint_dim = 0;
        for (auto dim : dims) { joint_dim += dim; }

        return joint_dim;
    }

    template <int...Indices>
    constexpr int expand_obsrv_dimension(IndexSequence<Indices...>) const
    {
        const auto& dims =
            { std::get<Indices>(models_)->observation_dimension()... };

        int joint_dim = 0;
        for (auto dim : dims) { joint_dim += dim; }

        return joint_dim;
    }

    template <
        int Size,
        int k = 0,
        typename Tuple,
        typename State,
        typename Noise,
        typename Obsrv
    >
    void predict_observation(Tuple& models_tuple,
                       double delta_time,
                       const State& state,
                       const Noise& noise,
                       Obsrv& prediction,
                       const int obsrv_offset = 0,
                       const int state_offset = 0,
                       const int noise_offset = 0)
    {
        auto&& model = std::get<k>(models_tuple);

        const auto obsrv_dim = model->observation_dimension();
        const auto state_dim = model->state_dimension();
        const auto noise_dim = model->noise_dimension();

        prediction.middleRows(obsrv_offset, obsrv_dim) =
            model->predict_observation(
                state.middleRows(state_offset, state_dim),
                noise.middleRows(noise_offset, noise_dim),
                delta_time
            );

        if (Size == k + 1) return;

        predict_observation<Size, k + (k + 1 < Size ? 1 : 0)>(
                    models_tuple,
                    delta_time,
                    state,
                    noise,
                    prediction,
                    obsrv_offset + obsrv_dim,
                    state_offset + state_dim,
                    noise_offset + noise_dim);
    }
    /** \endcond */
};

}

#endif

