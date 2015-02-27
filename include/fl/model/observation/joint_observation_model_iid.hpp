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
 * \file joint_observation_model_iid.hpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__MODEL__OBSERVATION__JOINT_OBSERVATION_MODEL_IID_HPP
#define FL__MODEL__OBSERVATION__JOINT_OBSERVATION_MODEL_IID_HPP

#include <Eigen/Dense>

#include <memory>

#include <fl/util/traits.hpp>
#include <fl/util/meta.hpp>
#include <fl/distribution/gaussian.hpp>

#include <fl/model/adaptive_model.hpp>
#include <fl/model/observation/observation_model_interface.hpp>

namespace fl
{

// Forward declarations
template <typename...Model> class JointObservationModel;

/**
 * Traits of JointObservationModel<MultipleOf<ObservationModel, Count>>
 */
template <
    typename ObservationModel,
    int Count
>
struct Traits<
           JointObservationModel<MultipleOf<ObservationModel, Count>>
        >
{
    enum : signed int { ModelCount = Count };

    typedef ObservationModel LocalObsrvModel;
    typedef typename Traits<ObservationModel>::Scalar Scalar;
    typedef typename Traits<ObservationModel>::State LocalState;
    typedef typename Traits<ObservationModel>::Noise LocalNoise;
    typedef typename Traits<ObservationModel>::Obsrv LocalObsrv;

    enum : signed int
    {
        ObsrvDim = ExpandSizes<LocalObsrv::SizeAtCompileTime, Count>::Size,        
        NoiseDim = ExpandSizes<LocalNoise::SizeAtCompileTime, Count>::Size,
        StateDim = LocalState::SizeAtCompileTime,
    };

    typedef Eigen::Matrix<Scalar, ObsrvDim, 1> Obsrv;
    typedef Eigen::Matrix<Scalar, StateDim, 1> State;
    typedef Eigen::Matrix<Scalar, NoiseDim, 1> Noise;

    typedef ObservationModelInterface<
                Obsrv,
                State,
                Noise
            > ObservationModelBase;
};

/**
 * \ingroup observation_models
 */
template <
    typename LocalObsrvModel,
    int Count>
class JointObservationModel<MultipleOf<LocalObsrvModel, Count>>
    : public Traits<
                 JointObservationModel<MultipleOf<LocalObsrvModel, Count>>
             >::ObservationModelBase
{
public:
    typedef JointObservationModel<MultipleOf<LocalObsrvModel,Count>> This;

    typedef typename Traits<This>::Obsrv Obsrv;
    typedef typename Traits<This>::State State;
    typedef typename Traits<This>::Noise Noise;

public:
    explicit JointObservationModel(
            const LocalObsrvModel& local_obsrv_model,
            int count = ToDimension<Count>::Value)
        : local_obsrv_model_(local_obsrv_model),
          count_(count)
    {
        assert(count_ > 0);
    }

    explicit
    JointObservationModel(const MultipleOf<LocalObsrvModel, Count>& mof)
        : local_obsrv_model_(mof.instance),
          count_(mof.count)
    {
        assert(count_ > 0);
    }

    /**
     * \brief Overridable default destructor
     */
    ~JointObservationModel() { }

    /**
     * \copydoc ObservationModelInterface::predict_obsrv
     */
    virtual Obsrv predict_obsrv(const State& state,
                                const Noise& noise,
                                double delta_time)
    {
        Obsrv y = Obsrv::Zero(obsrv_dimension(), 1);

        int obsrv_dim = local_obsrv_model_.obsrv_dimension();
        int noise_dim = local_obsrv_model_.noise_dimension();

        const int count = count_;
        for (int i = 0; i < count; ++i)
        {
            y.middleRows(i * obsrv_dim, obsrv_dim) =
                local_obsrv_model_.predict_obsrv(
                    state,
                    noise.middleRows(i * noise_dim, noise_dim),
                    delta_time);
        }

        return y;
    }

    virtual int obsrv_dimension() const
    {
        return local_obsrv_model_.obsrv_dimension() * count_;
    }

    virtual int noise_dimension() const
    {
        return local_obsrv_model_.noise_dimension() * count_;
    }

    virtual int state_dimension() const
    {
        return local_obsrv_model_.state_dimension();
    }

    LocalObsrvModel& local_observation_model()
    {
        return local_obsrv_model_;
    }

protected:
    LocalObsrvModel local_obsrv_model_;
    int count_;
};

/**
 * Traits of the \em Adaptive JointObservationModel
 */
template <
    typename ObsrvModel,
    int Count
>
struct Traits<
           JointObservationModel<MultipleOf<Adaptive<ObsrvModel>, Count>>
        >
    : Traits<JointObservationModel<MultipleOf<ObsrvModel, Count>>>
{
    typedef Traits<JointObservationModel<MultipleOf<ObsrvModel, Count>>> Base;

    typedef typename Traits<ObsrvModel>::Param LocalParam;

    enum : signed int
    {
        ParamDim = ExpandSizes<LocalParam::SizeAtCompileTime, Count>::Size
    };

    typedef Eigen::Matrix<typename Base::Scalar, ParamDim, 1> Param;

    typedef ObservationModelInterface<
                typename Base::Obsrv,
                typename Base::State,
                typename Base::Noise
            > ObservationModelBase;

    typedef AdaptiveModel<Param> AdaptiveModelBase;
};

/**
 * \ingroup observation_models
 *
 * JointObservationModel for adaptive local observation models
 */
template <
    typename LocalObsrvModel,
    int Count>
class JointObservationModel<MultipleOf<Adaptive<LocalObsrvModel>, Count>>
    : public Traits<
                 JointObservationModel<MultipleOf<Adaptive<LocalObsrvModel>, Count>>
             >::ObservationModelBase,
      public Traits<
                JointObservationModel<MultipleOf<Adaptive<LocalObsrvModel>, Count>>
             >::AdaptiveModelBase

{
public:
    typedef JointObservationModel<MultipleOf<Adaptive<LocalObsrvModel>, Count>> This;

    typedef typename Traits<This>::Param Param;

public:
    virtual Param param() const { return param_; }
    virtual void param(Param new_param) {  param_ = new_param; }
    virtual int param_dimension() const { return param_.rows(); }

protected:
    Param param_;
};

}

#endif

