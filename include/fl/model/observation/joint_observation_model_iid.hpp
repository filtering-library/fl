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
#include <type_traits>

#include <fl/util/traits.hpp>
#include <fl/util/meta.hpp>
#include <fl/distribution/gaussian.hpp>

#include <fl/model/adaptive_model.hpp>
#include <fl/model/observation/observation_model_interface.hpp>

namespace fl
{

// Forward declarations
template <typename...Models> class JointObservationModel;

/**
 * Traits of JointObservationModel<MultipleOf<ObservationModel, Count>>
 */
template <
    typename ObsrvModel,
    int Count
>
struct Traits<
           JointObservationModel<MultipleOf<ObsrvModel, Count>, Adaptive<>>
        >
{
    enum : signed int { ModelCount = Count };

    typedef ObsrvModel LocalObsrvModel;
    typedef typename Traits<ObsrvModel>::Scalar Scalar;
    typedef typename Traits<ObsrvModel>::Obsrv LocalObsrv;
    typedef typename Traits<ObsrvModel>::State LocalState;
    typedef typename Traits<ObsrvModel>::Param LocalParam;
    typedef typename Traits<ObsrvModel>::Noise LocalNoise;

    enum : signed int
    {
        StateDim = LocalState::SizeAtCompileTime,
        ObsrvDim = ExpandSizes<LocalObsrv::SizeAtCompileTime, Count>::Size,        
        NoiseDim = ExpandSizes<LocalNoise::SizeAtCompileTime, Count>::Size,
        ParamDim = ExpandSizes<LocalParam::SizeAtCompileTime, Count>::Size
    };

    typedef Eigen::Matrix<Scalar, ObsrvDim, 1> Obsrv;    
    typedef Eigen::Matrix<Scalar, StateDim, 1> State;
    typedef Eigen::Matrix<Scalar, NoiseDim, 1> Noise;
    typedef Eigen::Matrix<Scalar, ParamDim, 1> Param;

    typedef ObservationModelInterface<
                Obsrv,
                State,
                Noise
            > ObservationModelBase;
};

/**
 * \ingroup observation_models
 *
 * \brief JointObservationModel itself is an observation model which contains
 * internally multiple models all of the \em same type. The joint model can be
 * simply summarized as \f$ h(x, \theta, w) = [ h_{local}(x, \theta_1, w_1),
 * h_{local}(x, \theta_2, w_2), \ldots, h_{local}(x, \theta_n, w_n) ]^T \f$
 *
 * where \f$x\f$ is the state variate, \f$\theta = [ \theta_1, \theta_2, \ldots,
 * \theta_n ]^T\f$ and \f$w = [ w_1, w_2, \ldots, w_n ]^T\f$  are the the
 * parameters and noise terms of each of the \f$n\f$ models.
 *
 * JointObservationModel implements the ObservationModelInterface and the
 * AdaptiveModel interface. That being said, the JointObservationModel can be
 * used as a regular observation model or even as an adaptive observation model
 * which provides a set of parameters that can be changed at any time. This
 * implies that all sub-models must implement the the AdaptiveModel
 * interface. However, if the sub-model is not adaptive, JointObservationModel
 * applies a decorator call the NotAdaptive operator on the sub-model. This
 * operator enables any model to be treated as if it is adaptive without
 * effecting the model behaviour.
 */
template <
    typename LocalObsrvModel,
    int Count
>
#ifdef GENERATING_DOCUMENTATION
class JointObservationModel<MultipleOf<LocalObsrvModel, Count>>
#else
class JointObservationModel<MultipleOf<LocalObsrvModel, Count>, Adaptive<>>
#endif
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
 * \internal
 * \ingroup observation_models
 *
 * Forwards an adaptive LocalObsrvModel type to the JointObservationModel
 * implementation. \sa ForwardAdaptive for more details.
 */
template <
    typename LocalObsrvModel,
    int Count
>
class JointObservationModel<MultipleOf<LocalObsrvModel, Count>>
    : public JointObservationModel<
          MultipleOf<typename ForwardAdaptive<LocalObsrvModel>::Type, Count>,
          Adaptive<>
      >
{ };

}

#endif

