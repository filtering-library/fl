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
 * \file not_adaptive.hpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__UTIL__META__OPERATOR__NOT_ADAPTIVE_HPP
#define FL__UTIL__META__OPERATOR__NOT_ADAPTIVE_HPP

#include <Eigen/Dense>

#include <fl/util/traits.hpp>
#include <fl/util/meta/size_deduction.hpp>
#include <fl/model/adaptive_model.hpp>
#include <fl/model/process/process_model_interface.hpp>
#include <fl/model/observation/observation_model_interface.hpp>

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
    typedef typename Traits<Model>::Obsrv Obsrv;
    typedef typename Traits<Model>::State State;
    typedef typename Traits<Model>::Noise Noise;

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
 * \cond INTERNAL
 * NotAdaptive operator specialization forwarding to the according ModelType
 * (i.e. either internal::ObsrvModelType or internal::ProcessModelType).
 * \endcond
 */
template <typename Model>
class NotAdaptive<Model>
  : public NotAdaptive<typename Model::ModelType, Model>
{
public:
    typedef NotAdaptive<typename Model::ModelType, Model> Base;

    /**
     * Conversion constructor. This converts the actual model instance into
     * a NotAdaptive instances
     *
     * \param model     Actual model instance
     */
    NotAdaptive(const Model& model) : Base(model) { }
};

/**
 * \internal
 * \ingroup meta
 *
 * Traits of Non-Adaptive model
 */
template <typename Model>
struct Traits<NotAdaptive<internal::ObsrvModelType, Model>>
    : Traits<NotAdaptive<Model>>
{ };

/**
 * \internal
 * \ingroup meta
 *
 * Adaptive operator implementation of the observation model
 */
template <typename Model>
class NotAdaptive<internal::ObsrvModelType, Model>
  : public Model,
    public Traits<NotAdaptive<internal::ObsrvModelType, Model>>::AdaptiveModelBase
{
public:
    typedef NotAdaptive<internal::ObsrvModelType, Model> This;

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
    NotAdaptive(const Model& model) : Model(model) { }

    /**
     * Evaluates the model function \f$y = h(x, w)\f$ where \f$x\f$ is the state
     * and \f$w\sim {\cal N}(0, 1)\f$ is a white noise parameter. Put
     * differently, \f$y = h(x, w)\f$ is a sample from the conditional model
     * distribution \f$p(y \mid x)\f$.
     *
     * \param state         The state variable \f$x\f$
     * \param noise         The noise term \f$w\f$
     * \param delta_time    Prediction time
     */
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


//class NAProcessModel;

///**
// * Traits of ObservationModelInterface
// */
//template <>
//struct Traits<NAProcessModel>
//{
//    typedef internal::NullVector State;
//    typedef internal::NullVector Input;
//    typedef internal::NullVector Noise;
//    typedef internal::NullVector::Scalar Scalar;
//};

///**
// * \internal
// * \ingroup process_models
// *
// */
//class NAProcessModel
//    : public ProcessModelInterface<
//                 internal::NullVector,
//                 internal::NullVector,
//                 internal::NullVector>
//{
//public:
//    typedef NAProcessModel This;

//    typedef typename Traits<This>::State State;
//    typedef typename Traits<This>::Input Input;
//    typedef typename Traits<This>::Noise Noise;
//    typedef typename Traits<This>::Scalar Scalar;

//public:
//    virtual State predict_state(double,
//                                const State&,
//                                const Noise&,
//                                const Input&) { return State(); }
//    virtual constexpr size_t state_dimension() const { return 0; }
//    virtual constexpr size_t noise_dimension() const { return 0; }
//    virtual constexpr size_t input_dimension() const { return 0; }
//};

}

#endif
