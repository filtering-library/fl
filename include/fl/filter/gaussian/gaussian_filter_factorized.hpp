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
 * \file gaussian_filter_factorized.cpp
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
#include <fl/model/observation/joint_observation_model.hpp>


namespace fl
{

// Forward declarations
template <typename...> class GaussianFilter;

/**
 * \defgroup gaussian_filter_iid Factorized Gaussian Filter (IID)
 * \ingroup sigma_point_kalman_filters
 */

/** \cond INTERNAL */
#ifdef TemplateParameters
    #undef TemplateParameters
#endif

#define Template_Parameters\
        JointProcessModel<\
            StateProcessModel,\
            JointProcessModel<\
                MultipleOf<SingleParamProcessModel, Count>>>,\
        JointObservationModel<\
            MultipleOf<SingleObservationModel, Count>>,\
        PointSetTransform

/** \endcond */

/**
 * \ingroup gaussian_filter_iid
 * Traits of Factorized GaussianFilter for IID Parameters
 */
#ifndef GENERATING_DOCUMENTATION
template <
    typename StateProcessModel,
    typename LocalParamModel,
    typename LocalObsrvModel,
    int Count,
    typename PointSetTransform
>
#endif
struct Traits<
           GaussianFilter<
                JointProcessModel<
                    StateProcessModel,
                    JointProcessModel<MultipleOf<LocalParamModel, Count>>>,
                JointObservationModel<MultipleOf<LocalObsrvModel, Count>>,
                PointSetTransform,
                Options<FactorizeParams>>>
{
    /** \cond INTERNAL */
    /**
     * \brief Represents the factorized model of a set of independent parameters
     * which shall be filtered jointly with the state.
     */
    typedef JointProcessModel<
                MultipleOf<LocalParamModel, Count>
            > JointParamProcessModel;    

    /**
     * \brief Internal joint process model consisting of \c StateProcessModel
     * and the JointProcessModel of multiple ParamProcessModel.
     */
    typedef JointProcessModel<
                StateProcessModel,
                JointParamProcessModel
            > ProcessModel;

    /**
     * \brief Represents the joint sensor observation mode. The joint sensor
     * assumes that all sensor are of the same kind and have the same model.
     */
    typedef JointObservationModel<
                MultipleOf<LocalObsrvModel, Count>
            > ObservationModel;

    /**
     * \brief Marginal distribution of the state component
     */
    typedef Gaussian<
                typename Traits<ProcessModel>::State
            > StateMarginalDistribution;

    /**
     * \brief Marginal distribution of the parameter components. The marginal
     * ifself consist of multiple Gaussian marginals, one for each parameter.
     */
    typedef JointDistribution<
                MultipleOf<
                    Gaussian<typename Traits<LocalParamModel>::State>,
                    Count>
            > ParamMarginalDistribution;    
    /** \endcond */

    /**
     * \brief Final filter declaration
     */
    typedef GaussianFilter<
                ProcessModel,
                ObservationModel,
                PointSetTransform
            > Filter;

    /*
     * Required concept (interface) types
     *
     * - Ptr
     * - State
     * - Input
     * - Observation
     * - StateDistribution
     */
    //typedef std::shared_ptr<Filter> Ptr;
    typedef typename Traits<ProcessModel>::State State;
    typedef typename Traits<ProcessModel>::Input Input;
    typedef typename Traits<ObservationModel>::Obsrv Obsrv;

    /**
     * \brief This represents the joint state distribution.
     * The joint state distribution second centered moment is composed as
     * \f$\Sigma = \begin{bmatrix} \Sigma_a & 0 \\ 0 & \Sigma_b \end{bmatrix}\f$
     * where \f$\Sigma_a\f$ is the second moment of the marginal distribution of
     * the state component, and \f$\Sigma_b\f$ is the second moment of the
     * joint parameter moments
     * \f$ \Sigma_b = \begin{bmatrix}
     *                  \Sigma_{b_1} & 0 & 0 \\
     *                  0 & \ddots & 0 \\
     *                  0 & 0 & \Sigma_{b_n}
     *                \end{bmatrix}\f$
     */
    typedef JointDistribution<
                StateMarginalDistribution,
                ParamMarginalDistribution
            > StateDistribution;

    /** \cond INTERNAL */
    typedef typename Traits<ProcessModel>::State StateState;
    typedef typename Traits<ProcessModel>::Noise StateNoise;

    typedef typename Traits<LocalObsrvModel>::Param ParamState;
    typedef typename Traits<LocalObsrvModel>::Noise ParamNoise;
    typedef typename Traits<LocalObsrvModel>::Obsrv LocalObsrv;
    typedef typename Traits<LocalObsrvModel>::Noise LocalNoise;

    /**
     * \brief Represents the total number of points required by the point set
     * transform.
     */
    enum : signed int
    {
        NumberOfPoints = PointSetTransform::number_of_points(
                             JoinSizes<
                                 StateState::RowsAtCompileTime,
                                 StateNoise::RowsAtCompileTime,
                                 ParamState::RowsAtCompileTime,
                                 ParamNoise::RowsAtCompileTime,
                                 LocalNoise::RowsAtCompileTime
                             >::Size)
    };

    typedef PointSet<StateState, NumberOfPoints> StateStatePointSet;
    typedef PointSet<StateNoise, NumberOfPoints> StateNoisePointSet;
    typedef PointSet<ParamState, NumberOfPoints> ParamStatePointSet;
    typedef PointSet<ParamNoise, NumberOfPoints> ParamNoisePointSet;
    typedef PointSet<LocalNoise, NumberOfPoints> ObsrvNoisePointSet;

    typedef Eigen::Matrix<ParamStatePointSet, Count, 1> ParamStatePointSets;
    /** \endcond */
};


/**
 * \class GaussianFilter<TemplateParameters, FactorizeParameters>
 *
 * \ingroup gaussian_filter_iid
 *
 * This \c GaussianFilter represents a factorized implementation of a Sigma
 * Point Kalman Filter. The filter state consists of a coherent state component
 * and a factorized parameter component.
 *
 * \tparam JointProcessModel
 * \tparam JointObservationModel
 * \tparam PointSetTransform
 */
#ifndef GENERATING_DOCUMENTATION
template <
    typename StateProcessModel,
    typename LocalParamModel,
    typename LocalObsrvModel,
    int Count,
    typename PointSetTransform
>
#endif
class GaussianFilter<
          JointProcessModel<
              StateProcessModel,
              JointProcessModel<MultipleOf<LocalParamModel, Count>>>,
          JointObservationModel<MultipleOf<LocalObsrvModel, Count>>,
          PointSetTransform,
          Options<FactorizeParams>>
    :
    /* Implement the filter interface */
    public FilterInterface<
               GaussianFilter<
                    JointProcessModel<
                        StateProcessModel,
                        JointProcessModel<MultipleOf<LocalParamModel, Count>>>,
                    JointObservationModel<MultipleOf<LocalObsrvModel, Count>>,
                    PointSetTransform,
                    Options<FactorizeParams>>>
{
private:
    /** \cond INTERNAL */
    typedef GaussianFilter<
                JointProcessModel<
                    StateProcessModel,
                    JointProcessModel<MultipleOf<LocalParamModel, Count>>>,
                JointObservationModel<MultipleOf<LocalObsrvModel, Count>>,
                PointSetTransform,
                Options<FactorizeParams>
            > This;

    typedef typename Traits<This>::ProcessModel ProcessModel;
    typedef typename Traits<This>::ObservationModel ObservationModel;
    typedef typename Traits<This>::JointParamProcessModel JointParamProcessModel;

    typedef typename Traits<This>::StateStatePointSet StateStatePointSet;
    typedef typename Traits<This>::StateNoisePointSet StateNoisePointSet;
    typedef typename Traits<This>::ParamStatePointSet ParamStatePointSet;
    typedef typename Traits<This>::ParamNoisePointSet ParamNoisePointSet;
    typedef typename Traits<This>::ObsrvNoisePointSet ObsrvNoisePointSet;

    typedef typename Traits<This>::ParamStatePointSets ParamStatePointSets;
    /** \endcond */

protected:
    /** \cond INTERNAL */
    /**
     * \enum Variates
     */
    enum Variates
    {
        a = 0, /**< Coherent state vector component \f$a\f$ */
        v_a,   /**<  Noise vector state component \f$a\f$ */
        b,     /**< Joint vector of factorized parameter component $ */
        v_b,   /**< Joint noise vector of factorized parameters */
        b_i,   /**< Single parameter \f$b_i\f$ */
        v_b_i, /**< Noise vector of a singe parameter */
        y,     /**< Joint measurement */
        w,     /**< Joint measurement noise */
        y_i,   /**< Single measurement \f$y_i\f$ of \f$i\f$-th sensor */
        w_i    /**< Noise vector of the \f$i\f$-th sensor */
    };
    /** \endcond */

public:
    /* public concept interface types */
    typedef typename Traits<This>::State State;
    typedef typename Traits<This>::Input Input;
    typedef typename Traits<This>::Obsrv Obsrv;
    typedef typename Traits<This>::StateDistribution StateDistribution;

public:
    /**
     * Creates a factorized Gaussian filter
     *
     * \param state_process_model
     * \param parameter_process_model
     * \param obsrv_model
     * \param point_set_transform
     * \param parameter_count
     */
    GaussianFilter(
            const StateProcessModel& state_process_model,
            const LocalParamModel& param_process_model,
            const LocalObsrvModel& obsrv_model,
            const PointSetTransform& point_set_transform,
            int parameter_count = Count)
        : state_process_model_(state_process_model),
          param_process_model_(param_process_model),
          joint_param_process_model_(
              JointParamProcessModel(
                  param_process_model_,
                  parameter_count)),
          process_model_(state_process_model_, joint_param_process_model_),
          obsrv_model_(obsrv_model),
          point_set_transform_(point_set_transform)
    {
        assert(parameter_count > 0);
    }

    /**
     * \copydoc FilterInterface::predict
     */
    virtual void predict(double delta_time,
                         const Input& input,
                         const StateDistribution& prior_dist,
                         StateDistribution& predicted_dist)
    {
        const int marginal_dim = dim(a) + dim(v_a) +
                                 dim(b_i) + dim(v_b_i) +
                                 dim(w_i);

        (void) marginal_dim;

        //point_set_transform_->forward(prior_dist, );
    }

    /**
     * \copydoc FilterInterface::update
     */
    virtual void update(const Obsrv& y,
                        const StateDistribution& predicted_dist,
                        StateDistribution& posterior_dist)
    {

    }

    /**
     * \copydoc FilterInterface::predict_and_update
     */
    virtual void predict_and_update(double delta_time,
                                    const Input& input,
                                    const Obsrv& observation,
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

    const std::shared_ptr<JointParamProcessModel>& joint_param_process_model()
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

private:
    const int dim(int component) const
    {
        switch (component)
        {
        case a:     return state_process_model_.state_dimension();
        case v_a:   return obsrv_model_.noise_dimension();
        case b:     return joint_param_process_model_.state_dimension();
        case v_b:   return joint_param_process_model_.noise_dimension();
        case b_i:   return param_process_model_.state_dimension();
        case v_b_i: return param_process_model_.noise_dimension();
        case y:     return obsrv_model_.obsrv_dimension();
        case w:     return obsrv_model_.noise_dimension();
        case y_i:   return obsrv_model_.obsrv_dimension() / Count;
        case w_i:   return obsrv_model_.noise_dimension() / Count;

        default:
            // throw!
            throw Exception("Unknown variate component.");
        }
    }


private:
    StateProcessModel state_process_model_;
    LocalParamModel param_process_model_;
    JointParamProcessModel joint_param_process_model_;

    ProcessModel process_model_;
    ObservationModel obsrv_model_;
    PointSetTransform point_set_transform_;
};

template <
    typename StateProcessModel,
    typename ObsrvModel,
    typename ParamProcess,
    int Count,
    typename PointTransform
>
class GaussianFilter<
          StateProcessModel,
          Join<MultipleOf<Adaptive<ObsrvModel, ParamProcess>, Count>>,
          PointTransform,
          Options<FactorizeParams>>
    : public GaussianFilter<
                JointProcessModel<
                    StateProcessModel,
                    JointProcessModel<MultipleOf<ParamProcess, Count>>
                >,
                JointObservationModel<MultipleOf<ObsrvModel, Count>>,
                PointTransform,
                Options<FactorizeParams>>
{
public:
    typedef GaussianFilter<
                JointProcessModel<
                    StateProcessModel,
                    JointProcessModel<MultipleOf<ParamProcess, Count>>
                >,
                JointObservationModel<MultipleOf<ObsrvModel, Count>>,
                PointTransform,
                Options<FactorizeParams>
            > Base;


    template <typename LocalParamModel,
              typename LocalObsrvModel>
    GaussianFilter(
            const StateProcessModel& state_process_model,
            const LocalParamModel& param_process_model,
            const LocalObsrvModel& obsrv_model,
            const PointTransform& point_set_transform,
            int parameter_count = Count)
                : Base(state_process_model,
                       param_process_model,
                       obsrv_model,
                       point_set_transform,
                       parameter_count)
    { }
};

#ifdef Template_Parameters
    #undef Template_Parameters
#endif

}

#endif
