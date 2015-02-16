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
 * \file gaussian_filter_factorized.hpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__FILTER__GAUSSIAN__GAUSSIAN_FILTER_FACTORIZED_HPP
#define FL__FILTER__GAUSSIAN__GAUSSIAN_FILTER_FACTORIZED_HPP

#include <fl/util/meta.hpp>
#include <fl/util/traits.hpp>

#include <fl/exception/exception.hpp>
#include <fl/filter/filter_interface.hpp>
#include <fl/filter/gaussian/point_set.hpp>

#include <fl/distribution/gaussian.hpp>
#include <fl/distribution/joint_distribution.hpp>
#include <fl/model/process/joint_process_model.hpp>


namespace fl
{

// Forward declarations
template <typename...> class GaussianFilter;

/**
 * Traits of Factorized GaussianFilter for IID Parameters
 */
template <
    typename StateProcessModel,
    typename ParamProcessModel,
    int ParameterCount,
    typename ObservationModel,
    typename PointSetTransform
>
struct Traits<
    GaussianFilter<
        StateProcessModel,
        JointProcessModel<MultipleOf<ParamProcessModel, ParameterCount>>,
        ObservationModel,
        PointSetTransform>>
{
    /** \cond INTERNAL */
    /**
     * Represents the factorized model of a set of independent parameters
     * which shall be filtered jointly with the state.
     */
    typedef JointProcessModel<
                MultipleOf<ParamProcessModel, ParameterCount>
            > JointParameterProcessModel;

    /**
     * Internal joint process model consisting of \c StateProcessModel and
     * the JointProcessModel of multiple ParamProcessModel.
     */
    typedef JointProcessModel<
                StateProcessModel,
                JointParameterProcessModel
            > ProcessModel;

    /**
     * Filter declaration
     */
    typedef GaussianFilter<
                StateProcessModel,
                JointParameterProcessModel,
                ObservationModel,
                PointSetTransform
            > Filter;
    /** \endcond */

    /*
     * Required concept (interface) types
     *
     * - Ptr
     * - State
     * - Input
     * - Observation
     * - StateDistribution
     */
    typedef std::shared_ptr<Filter> Ptr;
    typedef typename Traits<ProcessModel>::State State;
    typedef typename Traits<ProcessModel>::Input Input;
    typedef typename Traits<ObservationModel>::Observation Observation;

    /**
     * This represents the joint state distribution
     *
     * The joint state distribution second centered moment is composed as
     *
     * \f$\Sigma = \begin{bmatrix} \Sigma_a & 0 \\ 0 & \Sigma_b \end{bmatrix}\f$
     *
     * where \f$\Sigma_a\f$ is the second moment of the marginal distribution of
     * the state component, and \f$\Sigma_b\f$ is the second moment of the
     * joint parameter moments
     *
     * \f$ \Sigma_b = \begin{bmatrix}
     *                  \Sigma_{b_1} & 0 & 0 \\
     *                  0 & \ddots & 0 \\
     *                  0 & 0 & \Sigma_{b_n}
     *                \end{bmatrix}\f$
     */
    typedef JointDistribution<
                Gaussian<typename Traits<StateProcessModel>::State>,
                JointDistribution<
                    MultipleOf<
                        Gaussian<typename Traits<ParamProcessModel>::State>,
                        ParameterCount
                    >
                >
            > StateDistribution;
};

/**
 * \ingroup sigma_point_kalman_filters
 *
 * This \c GaussianFilter represents a factorized implementation of a Sigma
 * Point Kalman Filter. The filter state consists of a coherent state component
 * and a factorized parameter component.
 *
 * \tparam StateProcessModel
 * \tparam ParamProcessModel
 * \tparam ParameterCount
 * \tparam ObservationModel
 * \tparam PointSetTransform
 */
template <
    typename StateProcessModel,
    typename ParamProcessModel,
    int ParameterCount,
    typename ObservationModel,
    typename PointSetTransform
>
class GaussianFilter<
          StateProcessModel,
          JointProcessModel<MultipleOf<ParamProcessModel, ParameterCount>>,
          ObservationModel,
          PointSetTransform
      >
    : /**
       * ProcessModelInterface Base
       */
      public Traits<
          GaussianFilter<
              StateProcessModel,
              JointProcessModel<MultipleOf<ParamProcessModel, ParameterCount>>,
              ObservationModel,
              PointSetTransform
          >
      >::ProcessModelBase
{
protected:
    /** \cond INTERNAL */
    typedef GaussianFilter<
                StateProcessModel,
                JointProcessModel<MultipleOf<ParamProcessModel, ParameterCount>>,
                ObservationModel,
                PointSetTransform
            > This;

    typedef typename Traits<This>::ProcessModel ProcessModel;
    typedef typename
        Traits<This>::JointParameterProcessModel JointParameterProcessModel;
    /** \endcond */

public:
    /* public concept interface types */
    typedef typename Traits<This>::State State;
    typedef typename Traits<This>::Input Input;
    typedef typename Traits<This>::Observation Observation;
    typedef typename Traits<This>::StateDistribution StateDistribution;

public:
    /**
     * Creates a factorized Gaussian filter
     *
     * @brief GaussianFilter
     * @param state_process_model
     * @param parameter_process_model
     * @param parameter_count
     */
    GaussianFilter(
            const std::shared_ptr<StateProcessModel>& state_process_model,
            const std::shared_ptr<ParamProcessModel>& parameter_process_model,
            const std::shared_ptr<ObservationModel> obsrv_model,
            const std::shared_ptr<PointSetTransform>& point_set_transform,
            int parameter_count = ToDimension<ParameterCount>::Value)
        : state_process_model_(state_process_model),
          joint_param_process_model_(
              std::make_shared<JointParameterProcessModel>(
                  parameter_process_model,
                  parameter_count)),
          process_model_(state_process_model_, joint_param_process_model_),
          obsrv_model_(obsrv_model),
          point_set_transform_(point_set_transform)
    {

    }

    /**
     * \copydoc FilterInterface::predict
     */
    virtual void predict(double delta_time,
                         const Input& input,
                         const StateDistribution& prior_dist,
                         StateDistribution& predicted_dist)
    {

    }


    /**
     * \copydoc FilterInterface::update
     */
    virtual void update(const Observation& y,
                        const StateDistribution& predicted_dist,
                        StateDistribution& posterior_dist)
    {

    }


    /**
     * \copydoc FilterInterface::predict_and_update
     */
    virtual void predict_and_update(double delta_time,
                                    const Input& input,
                                    const Observation& observation,
                                    const StateDistribution& prior_dist,
                                    StateDistribution& posterior_dist)
    {
        predict(delta_time, input, prior_dist, posterior_dist);
        update(observation, posterior_dist, posterior_dist);
    }


    const std::shared_ptr<StateProcessModel>& state_process_model()
    {
        return state_process_model_;
    }

    const std::shared_ptr<JointParameterProcessModel>&
    joint_parameter_process_model()
    {
        return joint_param_process_model_;
    }

    const std::shared_ptr<ProcessModel>& process_model()
    {
        return process_model_;
    }

    const std::shared_ptr<ObservationModel>& observation_model()
    {
        return obsrv_model_;
    }


    const std::shared_ptr<PointSetTransform>& point_set_transform()
    {
        return point_set_transform_;
    }

protected:
    std::shared_ptr<StateProcessModel> state_process_model_;
    std::shared_ptr<JointParameterProcessModel> joint_param_process_model_;

    std::shared_ptr<ProcessModel> process_model_;
    std::shared_ptr<ObservationModel> obsrv_model_;
    std::shared_ptr<PointSetTransform> point_set_transform_;
};

}
