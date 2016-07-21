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
 * \file joint_sensor_iid.hpp
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
#include <fl/model/sensor/interface/sensor_function.hpp>

namespace fl
{

// Forward declarations
template <typename...Models> class JointSensor;

/**
 * Traits of JointSensor<MultipleOf<Sensor, Count>>
 */
template <
    typename Sensor,
    int Count
>
struct Traits<JointSensor<MultipleOf<Sensor, Count>>>
{
    enum : signed int { ModelCount = Count };

    typedef Sensor LocalSensor;
    typedef typename Sensor::State State;
    typedef typename Sensor::Obsrv::Scalar Scalar;
    typedef typename Sensor::Obsrv LocalObsrv;
    typedef typename Sensor::Noise LocalNoise;

    enum : signed int
    {
        StateDim = SizeOf<State>::Value,
        ObsrvDim = ExpandSizes<SizeOf<LocalObsrv>::Value, Count>::Value,
        NoiseDim = ExpandSizes<SizeOf<LocalNoise>::Value, Count>::Value
    };

    typedef Eigen::Matrix<Scalar, ObsrvDim, 1> Obsrv;
    typedef Eigen::Matrix<Scalar, NoiseDim, 1> Noise;

    typedef SensorFunction<Obsrv, State, Noise> SensorFunctionBase;
};

/**
 * \ingroup sensors
 *
 * \brief JointSensor itself is an observation model which contains
 * internally multiple models all of the \em same type. The joint model can be
 * simply summarized as \f$ h(x, \theta, w) = [ h_{local}(x, \theta_1, w_1),
 * h_{local}(x, \theta_2, w_2), \ldots, h_{local}(x, \theta_n, w_n) ]^T \f$
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
 * interface. However, if the sub-model is not adaptive, JointSensor
 * applies a decorator call the NotAdaptive operator on the sub-model. This
 * operator enables any model to be treated as if it is adaptive without
 * effecting the model behaviour.
 */
template <
    typename LocalSensor,
    int Count
>
class JointSensor<MultipleOf<LocalSensor, Count>>
    : public Traits<
                 JointSensor<MultipleOf<LocalSensor, Count>>
             >::SensorFunctionBase,
      public Descriptor,
      private internal::JointSensorIidType
{
private:
    typedef JointSensor<MultipleOf<LocalSensor,Count>> This;

public:
    enum : signed int { ModelCount = Count };

    typedef LocalSensor LocalModel;
    typedef typename Traits<This>::LocalObsrv LocalObsrv;
    typedef typename Traits<This>::LocalNoise LocalNoise;

    typedef typename Traits<This>::Obsrv Obsrv;
    typedef typename Traits<This>::Noise Noise;
    typedef typename Traits<This>::State State;

public:
    JointSensor(
            const LocalSensor& local_sensor,
            int count = ToDimension<Count>::Value)
        : local_sensor_(local_sensor),
          count_(count)
    {
        assert(count_ > 0);
    }

    template <typename Model>
    JointSensor(const MultipleOf<Model, Count>& mof)
        : local_sensor_(mof.instance),
          count_(mof.count)
    {
        assert(count_ > 0);
    }

    /**
     * \brief Overridable default destructor
     */
    virtual ~JointSensor() noexcept { }

    Obsrv observation(const State& state, const Noise& noise) const override
    {
        Obsrv y = Obsrv::Zero(obsrv_dimension(), 1);

        const int obsrv_dim = local_sensor_.obsrv_dimension();
        const int noise_dim = local_sensor_.noise_dimension();

        for (int i = 0; i < count_; ++i)
        {
            local_sensor_.id(i);

            y.middleRows(i * obsrv_dim, obsrv_dim) =
                local_sensor_.observation(
                    state,
                    noise.middleRows(i * noise_dim, noise_dim));
        }

        return y;
    }

    int obsrv_dimension() const override
    {
        return local_sensor_.obsrv_dimension() * count_;
    }

    int noise_dimension() const override
    {
        return local_sensor_.noise_dimension() * count_;
    }

    int state_dimension() const override
    {
        return local_sensor_.state_dimension();
    }

    LocalSensor& local_sensor()
    {
        return local_sensor_;
    }

    const LocalSensor& local_sensor() const
    {
        return local_sensor_;
    }

    virtual std::string name() const
    {
        return "JointSensor<MultipleOf<"
                    + this->list_arguments(local_sensor_.name()) +
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
    mutable LocalSensor local_sensor_;
    int count_;
};

///**
// * Traits of JointSensor<MultipleOf<Sensor, Count>>
// */
//template <
//    typename Sensor,
//    int Count
//>
//struct Traits<
//           JointSensor<MultipleOf<Sensor, Count>>
//        >
//    : public Traits<
//                JointSensor<
//                    MultipleOf<
//                        typename ForwardAdaptive<Sensor>::Type, Count
//                    >,
//                    Adaptive<>>>
//{ };

///**
// * \internal
// * \ingroup sensors
// *
// * Forwards an adaptive LocalSensor type to the JointSensor
// * implementation. \sa ForwardAdaptive for more details.
// */
//template <
//    typename Sensor,
//    int Count
//>
//class JointSensor<MultipleOf<Sensor, Count>>
//    : public JointSensor<
//                MultipleOf<typename ForwardAdaptive<Sensor>::Type, Count>,
//                Adaptive<>>
//{
//public:
//    typedef JointSensor<
//                MultipleOf<typename ForwardAdaptive<Sensor>::Type, Count>,
//                Adaptive<>
//            > Base;

//    typedef typename ForwardAdaptive<Sensor>::Type ForwardedType;

//    JointSensor(
//            const Sensor& local_sensor,
//            int count = ToDimension<Count>::Value)
//        : Base(ForwardedType(local_sensor), count)
//    { }

//    JointSensor(const MultipleOf<Sensor, Count>& mof)
//        : Base(mof)
//    { }
//};


}


