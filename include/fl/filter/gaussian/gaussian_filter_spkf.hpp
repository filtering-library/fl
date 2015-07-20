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
 * \file gaussian_filter_spkf.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__FILTER__GAUSSIAN__GAUSSIAN_FILTER_SPKF_HPP
#define FL__FILTER__GAUSSIAN__GAUSSIAN_FILTER_SPKF_HPP


#include <Eigen/Dense>

#include <fl/util/meta.hpp>
#include <fl/util/traits.hpp>
#include <fl/util/profiling.hpp>
#include <fl/exception/exception.hpp>
#include <fl/distribution/gaussian.hpp>
#include <fl/filter/filter_interface.hpp>
#include <fl/filter/gaussian/transform/point_set.hpp>
#include <fl/filter/gaussian/quadrature/sigma_point_quadrature.hpp>
#include <fl/filter/gaussian/sigma_point_update_policy.hpp>
#include <fl/filter/gaussian/sigma_point_prediction_policy.hpp>
#include <fl/filter/gaussian/sigma_point_additive_update_policy.hpp>
#include <fl/filter/gaussian/sigma_point_additive_prediction_policy.hpp>

namespace fl
{

// Forward delcaration
template <typename...> class GaussianFilter;

/**
 * \internal
 *
 * GaussianFilter Traits
 */
template <
    typename StateTransitionModel,
    typename ObservationModel,
    typename Quadrature,
    typename ... Policies
>
struct Traits<
           GaussianFilter<
               StateTransitionModel, ObservationModel, Quadrature, Policies...>>
{
    typedef typename StateTransitionModel::State State;
    typedef typename StateTransitionModel::Input Input;
    typedef typename ObservationModel::Obsrv Obsrv;
    typedef Gaussian<State> Belief;
};

/**
 * \ingroup sigma_point_kalman_filters
 *
 * GaussianFilter represents all filters based on Gaussian distributed systems.
 * This includes the Kalman Filter and filters using non-linear models such as
 * Sigma Point Kalman Filter family.
 *
 * \tparam StateTransitionFunction
 * \tparam ObservationModel
 *
 */
template<
    typename StateTransitionFunction,
    typename ObservationFunction,
    typename Quadrature
>
class GaussianFilter<StateTransitionFunction, ObservationFunction, Quadrature>
    :
    /* Implement the filter interface */
#ifndef GENERATING_DOCUMENTATION
    public GaussianFilter<
               typename RemoveAdditivityOf<StateTransitionFunction>::Type,
               typename RemoveAdditivityOf<ObservationFunction>::Type,
               Quadrature,
               SigmaPointPredictPolicy<
                   Quadrature,
                   typename AdditivityOf<StateTransitionFunction>::Type>,
               SigmaPointUpdatePolicy<
                   Quadrature,
                   typename AdditivityOf<ObservationFunction>::Type>>
#else
    public GaussianFilter<
               StateTransitionFunction,
               ObservationFunction,
               Quadrature,
               PredictionPolicy,
               UpdatePolicy>
#endif
{
public:
    template <typename...Args>
    GaussianFilter(Args&& ... args)
        : GaussianFilter<
              typename RemoveAdditivityOf<StateTransitionFunction>::Type,
              typename RemoveAdditivityOf<ObservationFunction>::Type,
              Quadrature,
              SigmaPointPredictPolicy<
                  Quadrature,
                  typename AdditivityOf<StateTransitionFunction>::Type>,
              SigmaPointUpdatePolicy<
                  Quadrature,
                  typename AdditivityOf<ObservationFunction>::Type>>
          (std::forward<Args>(args)...)
    { }
};

/**
 * \ingroup sigma_point_kalman_filters
 *
 * GaussianFilter represents all filters based on Gaussian distributed systems.
 * This includes the Kalman Filter and filters using non-linear models such as
 * Sigma Point Kalman Filter family.
 *
 * \tparam StateTransitionFunction
 * \tparam ObservationModel
 *
 */
template<
    typename StateTransitionFunction,
    typename ObservationFunction,
    typename Quadrature,
    typename PredictionPolicy,
    typename UpdatePolicy
>
class GaussianFilter<
          StateTransitionFunction,
          ObservationFunction,
          Quadrature,
          PredictionPolicy,
          UpdatePolicy>
    :
    /* Implement the filter interface */
    public FilterInterface<
               GaussianFilter<
                   StateTransitionFunction,
                   ObservationFunction,
                   Quadrature,
                   PredictionPolicy,
                   UpdatePolicy>>,
    public Descriptor
{
public:
    typedef typename StateTransitionFunction::State State;
    typedef typename StateTransitionFunction::Input Input;
    typedef typename ObservationFunction::Obsrv Obsrv;
    typedef Gaussian<State> Belief;

public:
    /**
     * Creates a Gaussian filter
     *
     * \param process_model         Process model instance
     * \param obsrv_model           Obsrv model instance
     * \param transform   Point set tranfrom such as the unscented
     *                              transform
     */
    GaussianFilter(const StateTransitionFunction& process_model,
                   const ObservationFunction& obsrv_model,
                   const Quadrature& quadrature)
        : process_model_(process_model),
          obsrv_model_(obsrv_model),
          quadrature_(quadrature)
    { }

    /**
     * \brief Overridable default destructor
     */
    virtual ~GaussianFilter() { }

    /**
     * \copydoc FilterInterface::predict
     */
    virtual void predict(const Belief& prior_belief,
                         const Input& input,
                         Belief& predicted_belief)
    {
        prediction_policy_(process_model(),
                           quadrature(),
                           prior_belief,
                           input,
                           predicted_belief);
    }

    virtual void predict(const Belief& prior_belief,
                         const Input& input,
                         const long steps,
                         Belief& predicted_belief)
    {
        predicted_belief = prior_belief;

        for (int i = 0; i < steps; ++i)
        {
            prediction_policy_(process_model(),
                               quadrature(),
                               prior_belief,
                               input,
                               predicted_belief);
        }
    }

    /**
     * \copydoc FilterInterface::update
     */
    virtual void update(const Belief& predicted_belief,
                        const Obsrv& obsrv,
                        Belief& posterior_belief)
    {
        update_policy_(obsrv_model(),
                       quadrature(),
                       predicted_belief,
                       obsrv,
                       posterior_belief);
    }

    /**
     * \copydoc FilterInterface::predict_and_update
     */
    virtual void predict_and_update(const Belief& prior_belief,
                                    const Input& input,
                                    const Obsrv& observation,
                                    Belief& posterior_belief)
    {
        predict(prior_belief, input, posterior_belief);
        update(posterior_belief, observation, posterior_belief);
    }

public: /* factory functions */
    virtual Belief create_belief() const
    {
        // note: do not simplify!
        auto belief = Belief(process_model().state_dimension());
        return belief;
    }

public: /* accessors & mutators */
    StateTransitionFunction& process_model()
    {
        return process_model_;
    }

    ObservationFunction& obsrv_model()
    {
        return obsrv_model_;
    }

    Quadrature& quadrature()
    {
        return quadrature_;
    }

    const StateTransitionFunction& process_model() const
    {
        return process_model_;
    }

    const ObservationFunction& obsrv_model() const
    {
        return obsrv_model_;
    }

    const Quadrature& quadrature() const
    {
        return quadrature_;
    }


    virtual std::string name() const
    {
        return "GaussianFilter<"
                + list_arguments(
                       process_model().name(),
                       obsrv_model().name(),
                       quadrature().name(),
                       prediction_policy_.name(),
                       update_policy_.name())
                + ">";
    }

    virtual std::string description() const
    {
        return "Sigma point based GaussianFilter with"
                + list_descriptions(
                       process_model().description(),
                       obsrv_model().description(),
                       quadrature().description(),
                       prediction_policy_.description(),
                       update_policy_.description());
    }

protected:
    /** \cond internal */
    StateTransitionFunction process_model_;
    ObservationFunction obsrv_model_;
    Quadrature quadrature_;
    PredictionPolicy prediction_policy_;
    UpdatePolicy update_policy_;
    /** \endcond */
};

//#ifdef TEMPLATE_ARGUMENTS
//    #undef TEMPLATE_ARGUMENTS
//#endif


//#define TEMPLATE_ARGUMENTS \
//    JointProcessModel< \
//       ProcessModel, \
//       JointProcessModel<MultipleOf<LocalParamModel, Count>>>, \
//    Adaptive<JointObservationModel<MultipleOf<LocalObsrvModel, Count>>>, \
//    PointSetTransform,\
//    FeaturePolicy<>

//#ifndef GENERATING_DOCUMENTATION
//template <
//    typename ProcessModel,
//    typename LocalObsrvModel,
//    typename LocalParamModel,
//    int Count,
//    typename PointSetTransform,
//    template <typename...T> class FeaturePolicy
//>
//#endif
//struct Traits<
//           GaussianFilter<
//               ProcessModel,
//               Join<MultipleOf<Adaptive<LocalObsrvModel, LocalParamModel>, Count>>,
//               PointSetTransform,
//               FeaturePolicy<>,
//               Options<NoOptions>
//            >
//        >
//    : Traits<GaussianFilter<TEMPLATE_ARGUMENTS>>
//{ };

//#ifndef GENERATING_DOCUMENTATION
//template <
//    typename ProcessModel,
//    typename LocalObsrvModel,
//    typename LocalParamModel,
//    int Count,
//    typename PointSetTransform,
//    template <typename...T> class FeaturePolicy
//>
//#endif
//class GaussianFilter<
//          ProcessModel,
//          Join<MultipleOf<Adaptive<LocalObsrvModel, LocalParamModel>, Count>>,
//          PointSetTransform,
//          FeaturePolicy<>,
//          Options<NoOptions>
//      >
//    : public GaussianFilter<TEMPLATE_ARGUMENTS>
//{
//public:
//    typedef GaussianFilter<TEMPLATE_ARGUMENTS> Base;

//    GaussianFilter(
//        const ProcessModel& state_process_model,
//        const LocalParamModel& param_process_model,
//        const LocalObsrvModel& obsrv_model,
//        const PointSetTransform& transform,
//        const typename Traits<Base>::FeatureMapping& feature_mapping
//            = typename Traits<Base>::FeatureMapping(),
//        int parameter_count = Count)
//            : Base(
//                { state_process_model, {param_process_model, parameter_count} },
//                { obsrv_model, parameter_count },
//                transform,
//                feature_mapping)
//    { }
//};

//#undef TEMPLATE_ARGUMENTS

}

#endif
