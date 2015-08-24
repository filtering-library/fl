/*
 * This is part of the FL library, a C++ Bayesian filtering library
 * (https://github.com/filtering-library)
 *
 * Copyright (c) 2014 Jan Issac (jan.issac\gmail.com)
 * Copyright (c) 2014 Manuel Wuthrich (manuel.wuthrich\gmail.com)
 *
 * Max-Planck Institute for Intelligent Systems, AMD Lab
 * University of Southern California, CLMC Lab
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

#ifndef FL__MODEL_OBSERVATION__BODY_TAIL_OBSERVATION_MODEL_HPP
#define FL__MODEL_OBSERVATION__BODY_TAIL_OBSERVATION_MODEL_HPP

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

// Forward declation of BodyTailObsrvModel
template <
    typename BodyModel,
    typename TailModel
>
class BodyTailObsrvModel;

/**
 * \internal
 * Traits of BodyTailObsrvModel
 */
template <
    typename BodyModel,
    typename TailModel
>
struct Traits<BodyTailObsrvModel<BodyModel, TailModel>>
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

    typedef ObservationFunction<Obsrv, State, Noise> ObsrvFunction;
    typedef ObservationDensity<Obsrv, State> ObsrvDensity;

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
                  "BodyModel must implement ObservationFunction<...> interface");

    static_assert(std::is_base_of<
                    internal::NonAdditiveNoiseModelType,
                    TailModel>::value,
                  "TailModel must implement ObservationFunction<...> interface");

    static_assert(std::is_base_of<
                    ObservationDensity<Obsrv, State>,
                    BodyModel>::value,
                  "BodyModel must implement ObservationDensity<...> interface");

    static_assert(std::is_base_of<
                    ObservationDensity<Obsrv, State>,
                    TailModel>::value,
                  "TailModel must implement ObservationDensity<...> interface");
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
class BodyTailObsrvModel
    : public Traits<BodyTailObsrvModel<BodyModel, TailModel>>::ObsrvFunction,
      public Traits<BodyTailObsrvModel<BodyModel, TailModel>>::ObsrvDensity,
      public Descriptor
{
private:
    typedef BodyTailObsrvModel<BodyModel, TailModel> This;

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

public:
    /**
     * \brief Creates a BodyTailObsrvModel
     *
     * \param body      Body observation model
     * \param tail      Tail observation model
     * \param weight_threshold
     *                  \f$\in [0, 1]\f$ which determines the model selection.
     *                  Any value below the weight_threshold selects the tail,
     *                  otherwise the body is selected
     */
    BodyTailObsrvModel(const BodyModel& body,
                       const TailModel& tail,
                       Real weight_threshold = 0.1)
        : body_(body),
          tail_(tail),
          weight_threshold_(weight_threshold)
    {
        if (weight_threshold_ < Real(0) || weight_threshold_  > Real(1))
        {
            fl_throw(Exception("weight_threshold must be in [0; 1]"));
        }
    }

    /**
     * \brief Returns an observation prediction based on the provided state and
     *        noise variate.
     *
     * The result is either from the body or the tail model. The selection is
     * based on the last component of the noise variate. If it exceeds the
     * body-tail-model weight_threshold, the body model is evaluated, otherwise the
     * tail is selected. The noise variate size is give as specified by
     * noise_dimension()
     */
    Obsrv observation(const State& state, const Noise& noise) const override
    {
        assert(noise.size() == noise_dimension());

        // use the last noise component as a weight_threshold to select the model
        // since the noise is a standard normal variate, we transform it into
        // a uniformly distributed variate before weight_thresholding
        Real u = fl::normal_to_uniform(noise.bottomRows(1)(0));

        if(u > weight_threshold_)
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

        return (Real(1) - weight_threshold_) * prob_body +
                          weight_threshold_  * prob_tail;
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
     * \brief Returns the weight_threshold between body and tail model
     */
    Real weight_threshold() const
    {
        return weight_threshold_;
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

    /**
     * \brief Const reference to the tail observation model part
     */
    const TailModel& tail_model() const
    {
        return tail_;
    }

    virtual std::string name() const
    {
        return "BodyTailObsrvModel<"
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
     * \brief weight_threshold \f$\in [0, 1]\f$ which determines the model
     *        selection. Any value below the weight_threshold selects the tail,
     *        otherwise the body is selected.
     */
    Real weight_threshold_;

    /** \endcond */
};

}

#endif
