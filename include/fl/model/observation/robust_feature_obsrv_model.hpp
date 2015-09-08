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

#pragma once


#include <fl/util/meta.hpp>
#include <fl/util/traits.hpp>
#include <fl/util/types.hpp>
#include <fl/util/descriptor.hpp>
#include <fl/distribution/gaussian.hpp>
#include <fl/model/observation/interface/observation_density.hpp>
#include <fl/model/observation/interface/observation_function.hpp>

namespace fl
{

namespace internal
{

enum RobustFeatureDimension { RobustFeatureDimExt = 2 };

}

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
                         SizeOf<typename ObsrvModel::Obsrv>::Value,
                         internal::RobustFeatureDimExt
                     >::Value
                 >::Type,
                 typename ObsrvModel::State,
                 typename ObsrvModel::Noise>,
      public Descriptor

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
                             SizeOf<typename ObsrvModel::Obsrv>::Value,
                             internal::RobustFeatureDimExt
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
     * \param obsrv_model   Reference to the source observation model
     *
     * \note This model takes only a reference of to an existing lvalue of the
     *       source model
     */
    explicit RobustFeatureObsrvModel(ObsrvModel& obsrv_model)
        : obsrv_model_(obsrv_model),
          body_gaussian_(obsrv_model.obsrv_dimension())
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
        auto y = Obsrv(obsrv_dimension());

        auto weight = obsrv_model_.weight_threshold();
        auto prob_y = body_gaussian_.probability(input_obsrv);
        auto prob_tail = obsrv_model_
                            .tail_model()
                            .probability(input_obsrv, mean_state_);

        auto normalizer = 1.0 / ((1.0 - weight) * prob_y + weight * prob_tail);


        if(weight == 0)
        {
            y(0) = 0.0;
            if (internal::RobustFeatureDimExt == 2) y(1) = 1.0;
            y.bottomRows(obsrv_model_.obsrv_dimension()) = input_obsrv;
        }
        else if( !std::isfinite(1.0 / (weight * prob_tail)) )
        {
            std::cout << "prob_tail is too small: " << prob_tail <<
                      "     weight: " << weight <<
                         "   input_obsrv: " << input_obsrv.transpose()
                         << "     normalizer: " << normalizer << std::endl;
            exit(-1);
        }
        else if(std::isfinite(normalizer))
        {
            y(0) = weight * prob_tail;
            if (internal::RobustFeatureDimExt == 2) y(1) =
                                                    (1.0 - weight) *  prob_y;
            y.bottomRows(obsrv_model_.obsrv_dimension()) =
                                        (1.0 - weight) * prob_y * input_obsrv;
            y *= normalizer;
        }
        else // if the normalizer is not finite, we assume that the
        {
            std::cout << "normalizer in robust feature is not finite " <<
                      "     weight: " << weight <<
                         "   input_obsrv: " << input_obsrv.transpose()
                         << "     normalizer: " << normalizer << std::endl;
            exit(-1);

        }



        return y;
    }

    /**
     * \brief Sets the feature function \a mean_state;
     * \param mean_state        \f$ \mu_x \f$
     *
     * \todo PAPER REF
     */
    virtual void mean_state(const State& mean_state)
    {
        mean_state_ = mean_state;
    }

    /**
     * \brief Sets the feature function body moments
     * \param body_gaussian     \f${\cal N}(y_t\mid \mu_{y}, \Sigma_{yy})\f$
     *
     * \todo PAPER REF
     */
    virtual void body_moments(
        const typename FirstMomentOf<InputObsrv>::Type& mean_obsrv,
        const typename SecondMomentOf<InputObsrv>::Type& cov_obsrv)
    {
        body_gaussian_.mean(mean_obsrv);
        body_gaussian_.covariance(cov_obsrv);
    }

    /**
     * \brief Returns the dimension of the \a Obsrv which is dim(\a Obsrv) + 2
     */
    int obsrv_dimension() const override
    {
        return obsrv_model_.obsrv_dimension() + internal::RobustFeatureDimExt;
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


    virtual int id() const { return obsrv_model_.id(); }

    virtual void id(int new_id) { obsrv_model_.id(new_id); }

    virtual std::string name() const
    {
        return "RobustFeatureObsrvModel<"
                + this->list_arguments(embedded_obsrv_model().name())
                + ">";
    }

    virtual std::string description() const
    {
        return "Robust feature observation model with "
                + this->list_descriptions(embedded_obsrv_model().description());
    }

protected:
    /** \cond internal */

    /**
     * \brief Reference to the ource observation model
     */
    ObsrvModel& obsrv_model_;

    /**
     * \brief \f${\cal N}(y_t\mid \mu_{y}, \Sigma_{yy})\f$
     */
    Gaussian<InputObsrv> body_gaussian_;

    /**
     * \brief \f$\mu_x\f$
     */
    State mean_state_;

    /* \endcond */
};

}


