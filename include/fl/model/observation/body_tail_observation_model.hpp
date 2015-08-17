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
     * \f$1 + max(SizeOf<BodyModel>::Noise, SizeOf<BodyModel>::Noise)\f$
     */
    typedef typename VariateOfSize<
                JoinSizes<
                    MaxOf<
                        SizeOf< typename BodyModel::Noise >::Value,
                        SizeOf< typename TailModel::Noise >::Value
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
      public Traits<BodyTailObsrvModel<BodyModel, TailModel>>::ObsrvDensity
{
private:
    typedef BodyTailObsrvModel<BodyModel, TailModel> This;

public:
    typedef typename Traits<This>::Obsrv Obsrv;
    typedef typename Traits<This>::State State;
    typedef typename Traits<This>::Noise Noise;

public:
    /**
     * \brief Creates a BodyTailObsrvModel
     *
     * \param body      Body observation model
     * \param tail      Tail observation model
     * \param threshold Threshold \f$\in [0, 1]\f$ which determines the model
     *                  selection. Any value below the threshold selects the
     *                  tail, otherwise the body is selected
     */
    BodyTailObsrvModel(const BodyModel& body,
                       const TailModel& tail,
                       Real threshold)
        : body_(body),
          tail_(tail),
          threshold_(threshold)
    {
        if (threshold_ < Real(0) || threshold_  > Real(1))
        {
            fl_throw(Exception("Threshold must be in [0; 1]"));
        }
    }

    /**
     * \brief observation
     * \param state
     * \param noise
     * \return
     */
    Obsrv observation(const State& state, const Noise& noise) const override
    {
        assert(noise.size() == noise_dimension());

        // use the last noise component as a threshold to select the model
        // since the noise is a standard normal variate, we transform it into
        // a uniformly distributed variate before thresholding
        Real u = fl::normal_to_uniform(noise.bottomRows(1)(0));

        if(u > threshold_)
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
     * \brief probability
     * \param obsrv
     * \param state
     * \return
     */
    Real probability(const Obsrv& obsrv, const State& state) const override
    {
        Real prob_body = body_.probability(obsrv, state);
        Real prob_tail = tail_.probability(obsrv, state);

        return (Real(1) - threshold_) * prob_body + threshold_ * prob_tail;
    }

    /**
     * \brief log_probability
     * \param obsrv
     * \param state
     * \return
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
     * \f$1 + max(SizeOf<BodyModel>::Noise, SizeOf<BodyModel>::Noise)\f$
     */
    int noise_dimension() const override
    {
        return 1 + std::max(body_.noise_dimension(), tail_.noise_dimension());
    }

protected:
    BodyModel body_;
    TailModel tail_;
    Real threshold_;
};

}

#endif
