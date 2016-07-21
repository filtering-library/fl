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
 * \file gaussian_filter_nonlinear.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once


#include <fl/util/meta.hpp>
#include <fl/util/traits.hpp>
#include <fl/filter/gaussian/gaussian_filter_nonlinear_generic.hpp>
#include <fl/filter/gaussian/update_policy/sigma_point_update_policy.hpp>
#include <fl/filter/gaussian/update_policy/sigma_point_additive_update_policy.hpp>
#include <fl/filter/gaussian/update_policy/sigma_point_additive_uncorrelated_update_policy.hpp>
#include <fl/filter/gaussian/prediction_policy/sigma_point_additive_prediction_policy.hpp>
#include <fl/filter/gaussian/prediction_policy/sigma_point_prediction_policy.hpp>

namespace fl
{

/**
 * \defgroup nonlinear_gaussian_filter Nonlinear Gaussian Filter
 *
 * \ingroup generic_nonlinear_gaussian_filter
 */

/**
 * \ingroup nonlinear_gaussian_filter
 *
 * GaussianFilter represents all filters based on Gaussian distributed systems.
 * This includes the Kalman Filter and filters using non-linear models such as
 * Sigma Point Kalman Filter family.
 *
 * \tparam TransitionFunction
 * \tparam SensorFunction
 * \tparam Quadrature
 *
 */
template<
    typename TransitionFunction,
    typename SensorFunction,
    typename Quadrature
>
class GaussianFilter<TransitionFunction, SensorFunction, Quadrature>
    :
    /* Implement the filter interface */
#ifndef GENERATING_DOCUMENTATION
    public GaussianFilter<
               typename RemoveAdditivityOf<TransitionFunction>::Type,
               typename RemoveAdditivityOf<SensorFunction>::Type,
               Quadrature,
               SigmaPointPredictPolicy<
                   Quadrature,
                   typename AdditivityOf<TransitionFunction>::Type>,
               SigmaPointUpdatePolicy<
                   Quadrature,
                   typename AdditivityOf<SensorFunction>::Type>>
#else
    public GaussianFilter<
               TransitionFunction,
               SensorFunction,
               Quadrature,
               PredictionPolicy,
               UpdatePolicy>
#endif
{
public:
    GaussianFilter(const TransitionFunction& process_model,
                   const SensorFunction& obsrv_model,
                   const Quadrature& quadrature)
        : GaussianFilter<
              typename RemoveAdditivityOf<TransitionFunction>::Type,
              typename RemoveAdditivityOf<SensorFunction>::Type,
              Quadrature,
              SigmaPointPredictPolicy<
                  Quadrature,
                  typename AdditivityOf<TransitionFunction>::Type>,
              SigmaPointUpdatePolicy<
                  Quadrature,
                  typename AdditivityOf<SensorFunction>::Type>>
          (process_model, obsrv_model, quadrature)
    { }
};


//#ifdef TEMPLATE_ARGUMENTS
//    #undef TEMPLATE_ARGUMENTS
//#endif


//#define TEMPLATE_ARGUMENTS \
//    JointTransition< \
//       Transition, \
//       JointTransition<MultipleOf<LocalParamModel, Count>>>, \
//    Adaptive<JointSensor<MultipleOf<LocalSensor, Count>>>, \
//    PointSetTransform,\
//    FeaturePolicy<>

//#ifndef GENERATING_DOCUMENTATION
//template <
//    typename Transition,
//    typename LocalSensor,
//    typename LocalParamModel,
//    int Count,
//    typename PointSetTransform,
//    template <typename...T> class FeaturePolicy
//>
//#endif
//struct Traits<
//           GaussianFilter<
//               Transition,
//               Join<MultipleOf<Adaptive<LocalSensor, LocalParamModel>, Count>>,
//               PointSetTransform,
//               FeaturePolicy<>,
//               Options<NoOptions>
//            >
//        >
//    : Traits<GaussianFilter<TEMPLATE_ARGUMENTS>>
//{ };

//#ifndef GENERATING_DOCUMENTATION
//template <
//    typename Transition,
//    typename LocalSensor,
//    typename LocalParamModel,
//    int Count,
//    typename PointSetTransform,
//    template <typename...T> class FeaturePolicy
//>
//#endif
//class GaussianFilter<
//          Transition,
//          Join<MultipleOf<Adaptive<LocalSensor, LocalParamModel>, Count>>,
//          PointSetTransform,
//          FeaturePolicy<>,
//          Options<NoOptions>
//      >
//    : public GaussianFilter<TEMPLATE_ARGUMENTS>
//{
//public:
//    typedef GaussianFilter<TEMPLATE_ARGUMENTS> Base;

//    GaussianFilter(
//        const Transition& state_process_model,
//        const LocalParamModel& param_process_model,
//        const LocalSensor& obsrv_model,
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


