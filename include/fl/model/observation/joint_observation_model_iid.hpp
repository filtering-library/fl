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

#pragma once


#include <Eigen/Dense>

#include <memory>
#include <type_traits>

#include <fl/util/types.hpp>
#include <fl/util/traits.hpp>
#include <fl/util/descriptor.hpp>
#include <fl/util/meta.hpp>
#include <fl/distribution/gaussian.hpp>

#include <fl/model/adaptive_model.hpp>
#include <fl/model/observation/interface/observation_function.hpp>

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
struct Traits<JointObservationModel<MultipleOf<ObsrvModel, Count>>>
{
    enum : signed int { ModelCount = Count };

    typedef ObsrvModel LocalObsrvModel;
    typedef typename ObsrvModel::State State;
    typedef typename ObsrvModel::Obsrv::Scalar Scalar;
    typedef typename ObsrvModel::Obsrv LocalObsrv;
    typedef typename ObsrvModel::Noise LocalNoise;

    enum : signed int
    {
        StateDim = SizeOf<State>::Value,
        ObsrvDim = ExpandSizes<SizeOf<LocalObsrv>::Value, Count>::Value,
        NoiseDim = ExpandSizes<SizeOf<LocalNoise>::Value, Count>::Value
    };

    typedef Eigen::Matrix<Scalar, ObsrvDim, 1> Obsrv;
    typedef Eigen::Matrix<Scalar, NoiseDim, 1> Noise;

    typedef ObservationFunction<Obsrv, State, Noise> ObservationFunctionBase;
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
class JointObservationModel<MultipleOf<LocalObsrvModel, Count>>
    : public Traits<
                 JointObservationModel<MultipleOf<LocalObsrvModel, Count>>
             >::ObservationFunctionBase,
      public Descriptor,
      private internal::JointObservationModelIidType
{
private:
    typedef JointObservationModel<MultipleOf<LocalObsrvModel,Count>> This;

public:
    enum : signed int { ModelCount = Count };

    typedef LocalObsrvModel LocalModel;
    typedef typename Traits<This>::LocalObsrv LocalObsrv;
    typedef typename Traits<This>::LocalNoise LocalNoise;

    typedef typename Traits<This>::Obsrv Obsrv;
    typedef typename Traits<This>::Noise Noise;
    typedef typename Traits<This>::State State;

public:
    JointObservationModel(
            const LocalObsrvModel& local_obsrv_model,
            int count = ToDimension<Count>::Value)
        : local_obsrv_model_(local_obsrv_model),
          count_(count)
    {
        assert(count_ > 0);
    }

    template <typename Model>
    JointObservationModel(const MultipleOf<Model, Count>& mof)
        : local_obsrv_model_(mof.instance),
          count_(mof.count)
    {
        assert(count_ > 0);
    }

    /**
     * \brief Overridable default destructor
     */
    ~JointObservationModel() { }

    Obsrv observation(const State& state, const Noise& noise) const override
    {
        Obsrv y = Obsrv::Zero(obsrv_dimension(), 1);

        const int obsrv_dim = local_obsrv_model_.obsrv_dimension();
        const int noise_dim = local_obsrv_model_.noise_dimension();

        for (int i = 0; i < count_; ++i)
        {
            local_obsrv_model_.id(i);

            y.middleRows(i * obsrv_dim, obsrv_dim) =
                local_obsrv_model_.observation(
                    state,
                    noise.middleRows(i * noise_dim, noise_dim));
        }

        return y;
    }

    int obsrv_dimension() const override
    {
        return local_obsrv_model_.obsrv_dimension() * count_;
    }

    int noise_dimension() const override
    {
        return local_obsrv_model_.noise_dimension() * count_;
    }

    int state_dimension() const override
    {
        return local_obsrv_model_.state_dimension();
    }

    LocalObsrvModel& local_obsrv_model()
    {
        return local_obsrv_model_;
    }

    const LocalObsrvModel& local_obsrv_model() const
    {
        return local_obsrv_model_;
    }

    virtual std::string name() const
    {
        return "JointObservationModel<MultipleOf<"
                    + this->list_arguments(local_obsrv_model_.name()) +
               ", Count>>";
    }

    virtual std::string description() const
    {
        return "Joint observation model of multiple local observation models "
                " with non-additive noise.";
    }

    /**
     *
     * \brief Returns the number of local models within this joint model
     */
    virtual int count_local_models() const
    {
        return count_;
    }

protected:
    mutable LocalObsrvModel local_obsrv_model_;
    int count_;
};

///**
// * Traits of JointObservationModel<MultipleOf<ObservationModel, Count>>
// */
//template <
//    typename ObsrvModel,
//    int Count
//>
//struct Traits<
//           JointObservationModel<MultipleOf<ObsrvModel, Count>>
//        >
//    : public Traits<
//                JointObservationModel<
//                    MultipleOf<
//                        typename ForwardAdaptive<ObsrvModel>::Type, Count
//                    >,
//                    Adaptive<>>>
//{ };

///**
// * \internal
// * \ingroup observation_models
// *
// * Forwards an adaptive LocalObsrvModel type to the JointObservationModel
// * implementation. \sa ForwardAdaptive for more details.
// */
//template <
//    typename ObsrvModel,
//    int Count
//>
//class JointObservationModel<MultipleOf<ObsrvModel, Count>>
//    : public JointObservationModel<
//                MultipleOf<typename ForwardAdaptive<ObsrvModel>::Type, Count>,
//                Adaptive<>>
//{
//public:
//    typedef JointObservationModel<
//                MultipleOf<typename ForwardAdaptive<ObsrvModel>::Type, Count>,
//                Adaptive<>
//            > Base;

//    typedef typename ForwardAdaptive<ObsrvModel>::Type ForwardedType;

//    JointObservationModel(
//            const ObsrvModel& local_obsrv_model,
//            int count = ToDimension<Count>::Value)
//        : Base(ForwardedType(local_obsrv_model), count)
//    { }

//    JointObservationModel(const MultipleOf<ObsrvModel, Count>& mof)
//        : Base(mof)
//    { }
//};


}


