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
 * \file joint_observation_model_id.hpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once


#include <Eigen/Dense>

#include <tuple>
#include <memory>

#include <fl/util/traits.hpp>
#include <fl/util/meta.hpp>

#include <fl/model/observation/interface/observation_function.hpp>

namespace fl
{

// Forward declarations
template <typename...Models> class JointSensor;

/**
 * Traits of JointSensor
 */
template <typename...Models>
struct Traits<
           JointSensor<Models...>
       >
{
    enum : signed int
    {
        ObsrvDim = JoinSizes<Traits<Models>::Obsrv::SizeAtCompileTime...>::Size,
        NoiseDim = JoinSizes<Traits<Models>::Noise::SizeAtCompileTime...>::Size,
        ParamDim = JoinSizes<Traits<Models>::Param::SizeAtCompileTime...>::Size,
        StateDim = Traits<
                       typename FirstTypeIn<Models...>::Type
                    >::State::SizeAtCompileTime
    };

    typedef typename Traits<
                         typename FirstTypeIn<Models...>::Type
                     >::State::Scalar Scalar;

    typedef Eigen::Matrix<Scalar, ObsrvDim, 1> Obsrv;
    typedef Eigen::Matrix<Scalar, StateDim, 1> State;
    typedef Eigen::Matrix<Scalar, NoiseDim, 1> Noise;
    typedef Eigen::Matrix<Scalar, ParamDim, 1> Param;

    typedef SensorInterface<
                Obsrv,
                State,
                Noise
            > SensorBase;

    typedef AdaptiveModel<Param> AdaptiveModelBase;
};

/**
 * \ingroup observation_models
 *
 * \brief JointSensor itself is an observation model which contains
 * internally multiple models. Each of the models must operate on the same state
 * space. The joint model can be simply summarized as
 * \f$ h(x, \theta, w) = [ h_1(x, \theta_1, w_1), h_2(x, \theta_2, w_2),
 * \ldots, h_n(x, \theta_n, w_n) ]^T \f$
 *
 * where \f$x\f$ is the state variate, \f$\theta = [ \theta_1, \theta_2, \ldots,
 * \theta_n ]^T\f$ and \f$w = [ w_1, w_2, \ldots, w_n ]^T\f$  are the the
 * parameters and noise terms of each of the \f$n\f$ models.
 *
 * JointSensor implements the SensorInterface and the
 * AdaptiveModel interface. That being said, the JointSensor can be
 * used as a regular observation model or even as an adaptive observation model
 * which provides a set of parameters that can be changed at any time. This
 * implies that all sub-models must implement the the AdaptiveModel
 * interface. However, if a sub-model is not adaptive, JointSensor
 * applies a decorator call the NotAdaptive operator on the sub-model. This
 * operator enables any model to be treated as if it is adaptive without
 * effecting the model behaviour.
 */
template <typename ... Models>
class JointSensor
    : public Traits<JointSensor<Models...>>::SensorBase
{
private:
    /** Typdef of \c This for #from_traits(TypeName) helper */
    typedef JointSensor<Models...> This;

public:
    typedef from_traits(State);
    typedef from_traits(Noise);
    typedef from_traits(Obsrv);
    typedef from_traits(Param);

public:
    /**
     * Constructor a JointSensor which is a composition of mixture of
     * fixed and dynamic-size observation models.
     *
     * \param models    Variadic list of shared pointers of the models
     */
    explicit
    JointSensor(std::shared_ptr<Models>...models)
        : models_(models...)
    { }

    /**
     * \brief Overridable default destructor
     */
    virtual ~JointSensor() noexcept { }

    /**
     * \copydoc SensorInterface::predict_obsrv
     */
    virtual Obsrv predict_obsrv(const State& state,
                                            const Noise& noise,
                                            double delta_time)
    {
        Obsrv prediction = Obsrv(obsrv_dimension(), 1);

        predict_obsrv<sizeof...(Models)>(
            models_,
            delta_time,
            state,
            noise,
            prediction);

        return prediction;
    }

    /**
     * \copydoc SensorInterface::state_dimension
     *
     * Determines the joint size of the state vector of all models
     */
    virtual int state_dimension() const
    {
        return expand_state_dimension(CreateIndexSequence<sizeof...(Models)>());
    }

    /**
     * \copydoc SensorInterface::noise_dimension
     *
     * Determines the joint size of the noise vector of all models
     */
    virtual int noise_dimension() const
    {
        return expand_noise_dimension(CreateIndexSequence<sizeof...(Models)>());
    }

    /**
     * \copydoc SensorInterface::input_dimension
     *
     * Determines the joint size of the observation vector of all models
     */
    virtual int obsrv_dimension() const
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
    /** \cond internal */
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
            { std::get<Indices>(models_)->obsrv_dimension()... };

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
    void predict_obsrv(Tuple& models_tuple,
                       double delta_time,
                       const State& state,
                       const Noise& noise,
                       Obsrv& prediction,
                       const int obsrv_offset = 0,
                       const int state_offset = 0,
                       const int noise_offset = 0)
    {
        auto&& model = std::get<k>(models_tuple);

        const auto obsrv_dim = model->obsrv_dimension();
        const auto state_dim = model->state_dimension();
        const auto noise_dim = model->noise_dimension();

        prediction.middleRows(obsrv_offset, obsrv_dim) =
            model->predict_obsrv(
                state.middleRows(state_offset, state_dim),
                noise.middleRows(noise_offset, noise_dim),
                delta_time
            );

        if (Size == k + 1) return;

        predict_obsrv<Size, k + (k + 1 < Size ? 1 : 0)>(
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


