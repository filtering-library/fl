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
 * \file body_tail_observation_model.hpp
 * \date August 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once


#include <type_traits>
#include <fl/util/meta.hpp>
#include <fl/util/traits.hpp>
#include <fl/util/math.hpp>
#include <fl/exception/exception.hpp>
#include <fl/util/descriptor.hpp>
#include <fl/model/observation/interface/observation_function.hpp>
#include <fl/model/observation/interface/observation_density.hpp>

namespace fl
{

// Forward declation of BodyTailSensor
template <
    typename BodyModel,
    typename TailModel
>
class BodyTailSensor;

/**
 * \internal
 * Traits of BodyTailSensor
 */
template <
    typename BodyModel,
    typename TailModel
>
struct Traits<BodyTailSensor<BodyModel, TailModel>>
{
    /**
     * \brief Define the body-tail observation model \a Noise variate
     *
     * The noise variate size is
     * 1 + max(SizeOf<BodyModel>::Noise, SizeOf<BodyModel>::Noise)
     */
    typedef typename VariateOfSize<
                JoinSizes<
                    MaxOf<
                        SizeOf<typename BodyModel::Noise>::Value,
                        SizeOf<typename TailModel::Noise>::Value
                    >::Value,
                    1
                >::Value
            >::Type Noise;

    typedef typename BodyModel::Obsrv Obsrv;
    typedef typename BodyModel::State State;

    typedef SensorFunction<Obsrv, State, Noise> ObsrvFunction;
    typedef SensorDensity<Obsrv, State> ObsrvDensity;

    static_assert(std::is_same<
                    typename BodyModel::Obsrv,
                    typename TailModel::Obsrv>::value,
                  "Both models must be defined interms of the same Obsrv type");

    static_assert(std::is_same<
                    typename BodyModel::State,
                    typename TailModel::State>::value,
                  "Both models must be defined interms of the same State type");

    static_assert(std::is_base_of<
                    internal::NonAdditiveNoiseModelType,
                    BodyModel>::value,
                  "BodyModel must implement SensorFunction<...> interface");

    static_assert(std::is_base_of<
                    internal::NonAdditiveNoiseModelType,
                    TailModel>::value,
                  "TailModel must implement SensorFunction<...> interface");

    static_assert(std::is_base_of<
                    SensorDensity<Obsrv, State>,
                    BodyModel>::value,
                  "BodyModel must implement SensorDensity<...> interface");

    static_assert(std::is_base_of<
                    SensorDensity<Obsrv, State>,
                    TailModel>::value,
                  "TailModel must implement SensorDensity<...> interface");
};


/**
 * \ingroup observation_models
 *
 * \brief Represents an observation model mixture comprising a body model and
 * a tail model. The resulting model is a fat-tailed model. That is, the body
 * model becomes a fat-tailed model.
 */
template <
    typename BodyModel,
    typename TailModel
>
class BodyTailSensor
    : public Traits<BodyTailSensor<BodyModel, TailModel>>::ObsrvFunction,
      public Traits<BodyTailSensor<BodyModel, TailModel>>::ObsrvDensity,
      public Descriptor
{
private:
    typedef BodyTailSensor<BodyModel, TailModel> This;

public:
    /**
     * \brief Model observation \f$y\f$ type. This is the same as the
     * \c BodyModel::Obsrv and \c TailModel::Obsrv.
     */
    typedef typename Traits<This>::Obsrv Obsrv;

    /**
     * \brief Model state \f$x\f$ type. This is the same as the
     * \c BodyModel::State and \c TailModel::State.
     */
    typedef typename Traits<This>::State State;

    /**
     * \brief Model noise \f$w\f$ type. The noise variate size is given by
     * 1 + max(SizeOf<BodyModel>::Noise, SizeOf<BodyModel>::Noise)
     */
    typedef typename Traits<This>::Noise Noise;

    /**
     * \brief Represents the body observation function of this model
     */
    typedef BodyModel BodySensor;

    /**
     * \brief Represents the tail observation function of this model
     */
    typedef TailModel TailSensor;

public:
    /**
     * \brief Creates a BodyTailSensor
     *
     * \param body      Body observation model
     * \param tail      Tail observation model
     * \param tail_weight
     *                  \f$\in [0, 1]\f$ which determines the model selection.
     *                  Any value below the tail_weight selects the tail,
     *                  otherwise the body is selected
     */
    BodyTailSensor(const BodyModel& body,
                       const TailModel& tail,
                       Real tail_weight = 0.1)
        : body_(body),
          tail_(tail),
          tail_weight_(tail_weight)
    {
        if (tail_weight_ < Real(0) || tail_weight_  > Real(1))
        {
            fl_throw(Exception("tail_weight must be in [0; 1]"));
        }
    }

    virtual ~BodyTailSensor() noexcept { }

    /**
     * \brief Returns an observation prediction based on the provided state and
     *        noise variate.
     *
     * The result is either from the body or the tail model. The selection is
     * based on the last component of the noise variate. If it exceeds the
     * body-tail-model tail_weight, the body model is evaluated, otherwise the
     * tail is selected. The noise variate size is give as specified by
     * noise_dimension()
     */
    Obsrv observation(const State& state, const Noise& noise) const override
    {
        assert(noise.size() == noise_dimension());

        // use the last noise component as a tail_weight to select the model
        // since the noise is a standard normal variate, we transform it into
        // a uniformly distributed variate before tail_weighting
        Real u = fl::normal_to_uniform(noise.bottomRows(1)(0));

        if(u > tail_weight_)
        {
            auto noise_body = noise.topRows(body_.noise_dimension()).eval();
            auto y = body_.observation(state, noise_body);
            return y; // RVO
        }

        auto noise_tail = noise.topRows(tail_.noise_dimension()).eval();
        auto y = tail_.observation(state, noise_tail);
        return y; // RVO
    }

    /**
     * \brief Evalues the probability of the specified \a obsrv, i.e.
     * \f$p(y \mid x)\f$ where \f$y =\f$ \a obsrv and \f$x =\f$ \a state.
     *
     * \param obsrv     Observation \f$y\f$
     * \param state     State \f$x\f$
     *
     * The resulting probability is a mixture of the probabilities given by
     * the bode and tail models.
     */
    Real probability(const Obsrv& obsrv, const State& state) const override
    {
        Real prob_body = body_.probability(obsrv, state);
        Real prob_tail = tail_.probability(obsrv, state);

        return (Real(1) - tail_weight_) * prob_body +
                          tail_weight_  * prob_tail;
    }

    /**
     * \brief Evalues the log. probability of the specified \a obsrv, i.e.
     * \f$p(y \mid x)\f$ where \f$y =\f$ \a obsrv and \f$x =\f$ \a state.
     *
     * \param obsrv     Observation \f$y\f$
     * \param state     State \f$x\f$
     *
     * The resulting probability is a mixture of the probabilities given by
     * the bode and tail models.
     */
    Real log_probability(const Obsrv& obsrv, const State& state) const override
    {
        return std::log(probability(obsrv, state));
    }

    /**
     * \brief Returns the dimension of the measurement \f$h(x, w)\f$
     */
    int obsrv_dimension() const override
    {
        assert(body_.obsrv_dimension() == tail_.obsrv_dimension());
        return body_.obsrv_dimension();
    }

    /**
     * \brief Returns the dimension of the state variable \f$x\f$
     */
    int state_dimension() const override
    {
        assert(body_.state_dimension() == tail_.state_dimension());
        return body_.state_dimension();
    }

    /**
     * \brief Returns the dimension of the noise term \f$w\f$
     *
     * The noise variate size is
     * 1 + max(SizeOf<BodyModel>::Noise, SizeOf<BodyModel>::Noise)
     */
    int noise_dimension() const override
    {
        return 1 + std::max(body_.noise_dimension(), tail_.noise_dimension());
    }

    /**
     * \brief Returns the tail_weight between body and tail model
     */
    Real tail_weight() const
    {
        return tail_weight_;
    }

    /**
     * \brief Reference to the body observation model part
     */
    BodyModel& body_model()
    {
        return body_;
    }

    /**
     * \brief Reference to the tail observation model part
     */
    TailModel& tail_model()
    {
        return tail_;
    }

    /**
     * \brief Const reference to the body observation model part
     */
    const BodyModel& body_model() const
    {
        return body_;
    }


    virtual int id() const override { return body_.id(); }

    virtual void id(int new_id) override
    {
        body_.id(new_id);
        tail_.id(new_id);
    }

    /**
     * \brief Const reference to the tail observation model part
     */
    const TailModel& tail_model() const
    {
        return tail_;
    }

    virtual std::string name() const
    {
        return "BodyTailSensor<"
                + this->list_arguments(
                            body_model().name(),
                            tail_model().name())
                + ">";
    }

    virtual std::string description() const
    {
        return "Body-Tail- observation model with "
                + this->list_descriptions(
                            body_model().description(),
                            tail_model().description());
    }

protected:
    /** \cond internal */

    /**
     * \brief Body observation model
     */
    BodyModel body_;

    /**
     * \brief Tail observation model representing the fat-tail
     */
    TailModel tail_;

    /**
     * \brief tail_weight \f$\in [0, 1]\f$ which determines the model
     *        selection. Any value below the tail_weight selects the tail,
     *        otherwise the body is selected.
     */
    Real tail_weight_;

    /** \endcond */
};

}


