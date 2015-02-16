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

template <typename...> class GaussianFilter;


template <typename Filter> class JointGaussian;

template <typename Filter>
struct Traits<JointGaussian<Filter>>
{

};

/**
 * Traits of GaussianFilter<
 *             StateProcessModel,
 *             JointProcessModel<MultipleOf<ParamProcessModel, ParameterCount>>>
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
     * Internal joint process model consisting of \c StateProcessModel and
     * the JointProcessModel of multiple ParamProcessModel.
     */
    typedef JointProcessModel<
                StateProcessModel,
                JointProcessModel<MultipleOf<ParamProcessModel, ParameterCount>>
            > ProcessModel;

    typedef GaussianFilter<
                StateProcessModel,
                JointProcessModel<MultipleOf<ParamProcessModel, ParameterCount>>,
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
          PointSetTransform>
    : Traits<
          GaussianFilter<
              StateProcessModel,
              JointProcessModel<MultipleOf<ParamProcessModel, ParameterCount>>,
              ObservationModel,
              PointSetTransform
          >
      >::ProcessModelBase
{
public:

};

}
