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
 * \file robust_feature_obsrv_model.hpp
 * \date August 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__MODEL__OBSERVATION__ROBUST_FEATURE_OBSRV_MODEL_HPP
#define FL__MODEL__OBSERVATION__ROBUST_FEATURE_OBSRV_MODEL_HPP

#include <fl/util/meta.hpp>
#include <fl/util/traits.hpp>
#include <fl/util/types.hpp>
#include <fl/util/descriptor.hpp>
#include <fl/distribution/gaussian.hpp>
#include <fl/model/observation/interface/observation_density.hpp>
#include <fl/model/observation/interface/observation_function.hpp>

namespace fl
{

/**
 * \ingroup observation_models
 *
 * \brief Represents an observation function or model which takes an arbitrary
 *        observation model (one that implements an ObservationFunction & a
 *        an ObservationDensity) as an argument and maps it into a feature space
 *        used by the RobustGaussianFilter.
 */
template <typename ObsrvModel>
class RobustFeatureObsrvModel
    : public ObservationFunction<
                 /* Obsrv type of the size SizeOf<ObsrvModel::Obsrv> + 2 */
                 typename VariateOfSize<
                     JoinSizes<
                         SizeOf<typename ObsrvModel::Obsrv>::Value, 2
                     >::Value
                 >::Type,
                 typename ObsrvModel::State,
                 typename ObsrvModel::Noise>

{
private:
    typedef RobustFeatureObsrvModel<ObsrvModel> This;

public:
    /**
     * \brief \a InputObsrv type which is the same as
     *        ObsrvModel::Obsrv. \a InputObsrv is mapped into the feature space.
     *        The resulting type is \a Obsrv.
     */
    typedef typename ObsrvModel::Obsrv InputObsrv;

    /**
     * \brief \a Obsrv (\f$y_t\f$) type which is a variate of the size
     *        SizeOf<InputObsrv> + 2. \a Obsrv reside in the feature space.
     */
    typedef typename VariateOfSize<
                         JoinSizes<
                             SizeOf<typename ObsrvModel::Obsrv>::Value, 2
                         >::Value
                     >::Type Obsrv;

    /**
     * \brief \a State (\f$x_t\f$) type which is the same as ObsrvModel::State
     */
    typedef typename ObsrvModel::State State;

    /**
     * \brief \a Noise (\f$x_t\f$) type which is the same as ObsrvModel::Noise
     */
    typedef typename ObsrvModel::Noise Noise;

public:
    /**
     * \brief Constructs a robust feature observation model for the robust
     *        gaussian filter
     *
     * \param obsrv_model   Source observation model
     */
    explicit RobustFeatureObsrvModel(const ObsrvModel& obsrv_model)
        : obsrv_model_(obsrv_model)
    { }

    /**
     * \brief Overridable default destructor
     */
    virtual ~RobustFeatureObsrvModel() { }

    /**
     * \brief observation Returns a feature mapped observation
     */
    Obsrv observation(const State& state, const Noise& noise) const override
    {
        Obsrv y = feature_obsrv(obsrv_model_.observation(state, noise));
        return y; // RVO
    }

    /**
     * \brief Computes the robust feature given an input feature fron the
     *        source observation model
     */
    virtual Obsrv feature_obsrv(const InputObsrv& input_obsrv) const
    {
        Obsrv y(obsrv_dimension());

        Real prob_body = body_gaussian_.probability(input_obsrv);
        Real prob_tail = obsrv_model_.tail_model().probability(input_obsrv);
        Real weight = obsrv_model_.weight_threshold();

        Real normalizer = (Real(1) - weight) * prob_body + weight * prob_tail;

        y(0) = prob_body;
        y(1) = prob_body;
        y.bottomRows(obsry_model_.obsrv_dimension()) = prob_body * input_obsrv;

        y /= normalizer;

        return y;
    }

    /**
     * \brief Sets the feature function (feature observation modek) and
     *        parameters
     * \param body_gaussian     \f${\cal N}(y_t\mid \mu_{y}, \Sigma_{yy})\f$
     * \param mean_state        \f$ \mu_x \f$
     *
     * PAPER REF
     */
    virtual void parameter(const Gaussian<InputObsrv>& body_gaussian,
                           const State& mean_state)
    {
        body_gaussian_ = body_gaussian;
        mean_state_ = mean_state;
    }

    /**
     * \brief Returns the dimension of the \a Obsrv which is dim(\a Obsrv) + 2
     */
    int obsrv_dimension() const override
    {
        return obsrv_model_.obsrv_dimension() + 2;
    }

    int noise_dimension() const override
    {
        return obsrv_model_.noise_dimension();
    }

    int state_dimension() const override
    {
        return obsrv_model_.state_dimension();
    }

    ObsrvModel& embedded_obsrv_model()
    {
        return obsrv_model_;
    }

    const ObsrvModel& embedded_obsrv_model() const
    {
        return obsrv_model_;
    }

protected:
    /* \cond internal */

    /**
     * \brief obsrv_model_ source observation model
     */
    ObsrvModel obsrv_model_;

    /**
     * \brief \f${\cal N}(y_t\mid \mu_{y}, \Sigma_{yy})\f$
     */
    Gaussian<InputObsrv> body_gaussian_;

    /**
     * \brief \f$ \mu_x \f$
     */
    State mean_state_;

    /* \endcond */
};

}

#endif
