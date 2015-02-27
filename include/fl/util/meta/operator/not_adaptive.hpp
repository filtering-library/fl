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
 * \internal
 * \ingroup meta
 *
 * MakeAdaptive operator specialization applied on the passed model type
 */
template <typename Model>
class NotAdaptive<Model>
  : public NotAdaptive<typename Model::ModelType, Model>
{ };

/**
 * \ingroup meta
 *
 * Traits of Non-Adaptive model
 */
template <typename Model>
struct Traits<NotAdaptive<internal::ObsrvModelType, Model>>
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
 * \internal
 * \ingroup meta
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

}

#endif
