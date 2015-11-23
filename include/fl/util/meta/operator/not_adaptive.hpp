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
 * \file not_adaptive.hpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once


#include <type_traits>

#include <Eigen/Dense>

#include <fl/util/traits.hpp>
#include <fl/util/meta.hpp>
#include <fl/model/adaptive_model.hpp>
#include <fl/model/process/interface/state_transition_function.hpp>
#include <fl/model/observation/interface/observation_function.hpp>

namespace fl
{

/**
 * \ingroup meta
 *
 * Not-Adaptive operator
 */
template <typename... Model> class NotAdaptive;

/**
 * \ingroup meta
 *
 * Traits of Non-Adaptive model
 */
template <typename Model>
struct Traits<NotAdaptive<Model>>
    : Traits<Model>
{
    typedef typename Traits<Model>::Scalar Scalar;

    enum : signed int { ParamDim = 0 };
    typedef Eigen::Matrix<Scalar, ParamDim, 1> Param;
    typedef AdaptiveModel<Param> AdaptiveModelBase;
};

/**
 * \ingroup meta
 *
 * \brief The NotAdaptive operator decorates any model such that it represents a
 * non-adaptive model. This operator can be applied to any model inparticular
 * to models which do not implement the AdaptiveModel interface.
 *
 * The way this operates is simply by overriding the AdaptiveModel interface
 * and deactivating it. If the model does not implement the AdaptiveModel
 * interface, then the NotAdaptive injects the interface with a deactivated
 * adaptivity.
 *
 * \cond internal
 * NotAdaptive operator specialization forwarding to the according ModelType
 * (i.e. either internal::ObsrvModelType or internal::ProcessModelType).
 * \endcond
 */
template <typename Model>
class NotAdaptive<Model>
  : public NotAdaptive<
               Model,
               typename Model::ModelType,
               Options<
                   std::is_base_of<internal::AdaptiveModelType, Model>::value
                >
            >
{
public:
    typedef NotAdaptive<
                Model,
                typename Model::ModelType,
                Options<
                    std::is_base_of<internal::AdaptiveModelType, Model>::value
                >
            > Base;

    /**
     * Conversion constructor. This converts the actual model instance into
     * a NotAdaptive instances
     *
     * \param model     Actual model instance
     */
    template <typename ... M>
    NotAdaptive(const M& ... model) : Base(model...) { }
};

/**
 * \internal
 * \ingroup meta
 *
 * Traits of Non-Adaptive observation models
 */
template <typename Model, int Adaptivity>
struct Traits<
           NotAdaptive<Model, internal::ObsrvModelType, Options<Adaptivity>>
       >
    : Traits<NotAdaptive<Model>>
{ };

/**
 * \internal
 * \ingroup meta
 *
 * Traits of Non-Adaptive process model
 */
template <typename Model, int Adaptivity>
struct Traits<
           NotAdaptive<Model, internal::ProcessModelType, Options<Adaptivity>>
        >
    : Traits<NotAdaptive<Model>>
{ };


/**
 * \internal
 * \ingroup meta
 *
 * NotAdaptive operator implementation of the observation model
 */
template <typename Model>
class NotAdaptive<Model, internal::ObsrvModelType, Options<IsNotAdaptive>>
  : public Model,
    public Traits<
               NotAdaptive<
                   Model,
                   internal::ObsrvModelType,
                   Options<IsNotAdaptive>
               >
           >::AdaptiveModelBase
{
public:
    typedef NotAdaptive<
                Model,
                internal::ObsrvModelType,
                Options<IsNotAdaptive>
            > This;

    typedef typename Traits<This>::Obsrv Obsrv;
    typedef typename Traits<This>::State State;
    typedef typename Traits<This>::Noise Noise;
    typedef typename Traits<This>::Param Param;

    /**
     * Conversion constructor. This converts the actual model instance into
     * a NotAdaptive instances
     *
     * \param model     Actual model instance
     */
    template <typename ... M>
    NotAdaptive(const M& ... model) : Model(model...) { }

    virtual Obsrv predict_obsrv(const State& state,
                                const Noise& noise,
                                double delta_time)
    {
        assert(state.rows() == state_dimension());

        return Model::predict_obsrv(state.topRows(Model::state_dimension()),
                                    noise, delta_time);
    }

    virtual int state_dimension() const
    {
        return Model::state_dimension() + param_dimension();
    }

    virtual void param(Param) {  }
    virtual const Param& param() const { return param_; }
    virtual int param_dimension() const { return 0; }

protected:
    Param param_;
};

/**
 * \internal
 * \ingroup meta
 *
 * NotAdaptive operator implementation of the observation model
 */
template <typename Model>
class NotAdaptive<Model, internal::ObsrvModelType, Options<IsAdaptive>>
  : public Model
{
public:
    typedef NotAdaptive<
                Model,
                internal::ObsrvModelType,
                Options<IsAdaptive>
            > This;

    typedef typename Traits<This>::Obsrv Obsrv;
    typedef typename Traits<This>::State State;
    typedef typename Traits<This>::Noise Noise;
    typedef typename Traits<This>::Param Param;

    /**
     * Conversion constructor. This converts the actual model instance into
     * a NotAdaptive instances
     *
     * \param model     Actual model instance
     */
    template <typename ... M>
    NotAdaptive(const M& ... model) : Model(model...) { }

    virtual Obsrv predict_obsrv(const State& state,
                                const Noise& noise,
                                double delta_time)
    {
        assert(state.rows() == state_dimension());

        return Model::predict_obsrv(state.topRows(Model::state_dimension()),
                                    noise, delta_time);
    }

    virtual int state_dimension() const
    {
        return Model::state_dimension() + param_dimension();
    }

    virtual void param(Param) {  }
//    virtual const Param& param() const { return param_; }
    virtual int param_dimension() const { return 0; }

protected:
    Param param_;
};

/**
 * \internal
 * \ingroup meta
 *
 * NotAdaptive operator implementation of the process model
 */
template <typename Model>
class NotAdaptive<Model, internal::ProcessModelType, Options<IsNotAdaptive>>
  : public Model,
    public Traits<
               NotAdaptive<
                   Model,
                   internal::ProcessModelType,
                   Options<IsNotAdaptive>>
           >::AdaptiveModelBase
{
public:
    typedef NotAdaptive<
                Model,
                internal::ProcessModelType,
                Options<IsNotAdaptive>
            > This;

    typedef typename Traits<This>::Input Input;
    typedef typename Traits<This>::State State;
    typedef typename Traits<This>::Noise Noise;
    typedef typename Traits<This>::Param Param;

    /**
     * Conversion constructor. This converts the actual model instance into
     * a NotAdaptive instances
     *
     * \param model     Actual model instance
     */
    template <typename ... M>
    NotAdaptive(const M& ... model) : Model(model...) { }

    virtual State predict_state(double delta_time,
                                const State& state,
                                const Noise& noise,
                                const Input& input)
    {
        assert(state.rows() == state_dimension());

        return Model::predict_state(
                    delta_time,
                    state.topRows(Model::state_dimension()),
                    noise,
                    input);
    }

    virtual int state_dimension() const
    {
        return Model::state_dimension() + param_dimension();
    }

    virtual void param(Param) {  }
    virtual const Param& param() const { return param_; }
    virtual int param_dimension() const { return 0; }

protected:
    Param param_;
};



/**
 * \internal
 * \ingroup meta
 *
 * NotAdaptive operator implementation of the process model
 */
template <typename Model>
class NotAdaptive<Model, internal::ProcessModelType, Options<IsAdaptive>>
  : public Model
{
public:
    typedef NotAdaptive<
                Model,
                internal::ProcessModelType,
                Options<IsAdaptive>
            > This;

    typedef typename Traits<This>::Input Input;
    typedef typename Traits<This>::State State;
    typedef typename Traits<This>::Noise Noise;
    typedef typename Traits<This>::Param Param;

    /**
     * Conversion constructor. This converts the actual model instance into
     * a NotAdaptive instances
     *
     * \param model     Actual model instance
     */
    template <typename ... M>
    NotAdaptive(const M& ... model) : Model(model...) { }

    virtual State predict_state(double delta_time,
                                const State& state,
                                const Noise& noise,
                                const Input& input)
    {
        assert(state.rows() == state_dimension());

        return Model::predict_state(
                    delta_time,
                    state.topRows(Model::state_dimension()),
                    noise,
                    input);
    }

    virtual int state_dimension() const
    {
        return Model::state_dimension() + param_dimension();
    }

    virtual void param(Param) {  }
//    virtual const Param& param() const { return param_; }
    virtual int param_dimension() const { return 0; }

protected:
    Param param_;
};

}


