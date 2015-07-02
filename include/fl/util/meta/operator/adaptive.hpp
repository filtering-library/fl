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
 * \file adaptive.hpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__UTIL__META__OPERATOR__ADAPTIVE_HPP
#define FL__UTIL__META__OPERATOR__ADAPTIVE_HPP

#include <fl/util/meta/size_deduction.hpp>
#include <fl/util/meta/operator/not_adaptive.hpp>
#include <fl/model/adaptive_model.hpp>
#include <fl/model/process/interface/process_model_interface.hpp>
#include <fl/model/observation/interface/observation_model_interface.hpp>

namespace fl
{

/**
 * \ingroup meta
 *
 * Adaptive operator
 */
template <typename... Model> class Adaptive { };

/**
 * \ingroup meta
 */
template <typename Model>
struct Traits<Adaptive<Model>>
    : Traits<Model>
{
    typedef typename Traits<Model>::State ActualState;

    typedef Eigen::Matrix<
                typename Traits<Model>::Scalar,
                JoinSizes<
                    ActualState::SizeAtCompileTime,
                    Traits<Model>::Param::SizeAtCompileTime
                >::Size,
                1
            > State;
};

/**
 * \internal
 * \ingroup meta
 *
 * Adaptive operator specialization applied on the passed model type
 */
template <typename Model>
class Adaptive<Model>
  : public Adaptive<typename Model::ModelType, Model>
{
public:
    typedef Adaptive<typename Model::ModelType, Model> Base;

    template <typename...Args>
    Adaptive(Args&&...args) : Base(std::forward<Args>(args)...) { }
};

/**
 * \internal
 * \ingroup meta
 */
template <typename Model>
class Adaptive<NotAdaptive<Model>>
  : public Adaptive<
               typename Model::ModelType,
               NotAdaptive<typename Model::ModelType, Model>
           >
{
public:
    typedef Adaptive<
                typename Model::ModelType,
                NotAdaptive<typename Model::ModelType, Model>
            > Base;

    template <typename...Args>
    Adaptive(Args&&...args) : Base(std::forward<Args>(args)...) { }
};

/**
 * \ingroup meta
 */
template <typename Model>
struct Traits<Adaptive<internal::ObsrvModelType, Model>>
    : Traits<Adaptive<Model>>
{
};

/**
 * Adaptive operator implementation of the observation model
 */
template <typename Model>
class Adaptive<internal::ObsrvModelType, Model>
  : public Model
{
public:
    typedef Adaptive This;

    typedef typename Traits<This>::State State;

    typedef typename Traits<Model>::Obsrv Obsrv;
    typedef typename Traits<Model>::Noise Noise;
    typedef typename Traits<Model>::Param Param;

    template <typename...Args>
    Adaptive(Args&&...args) : Model(std::forward<Args>(args)...) { }

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
        assert(state.rows() ==
               Model::param_dimension() + Model::state_dimension());

        Model::param(state.bottomRows(Model::param_dimension()));

        return Model::predict_obsrv(state.topRows(Model::state_dimension()),
                                    noise, delta_time);
    }

    virtual int state_dimension() const
    {
        return Model::state_dimension() + Model::param_dimension();
    }
};

}

#endif
