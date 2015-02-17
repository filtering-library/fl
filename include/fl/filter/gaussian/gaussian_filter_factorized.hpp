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
 * \defgroup gaussian_filter_iid Factorized Gaussian Filter (IID)
 * \ingroup sigma_point_kalman_filters
 */

/**
 * \ingroup gaussian_filter_iid
 * Traits of Factorized GaussianFilter for IID Parameters
 */
template <
    typename StateProcessModel,
    typename ParamProcessModel,
    typename ObservationModel,
    int SensorCount,
    typename PointSetTransform
>
struct Traits<
           GaussianFilter<
               StateProcessModel,
               JointProcessModel<MultipleOf<ParamProcessModel, SensorCount>>,
               JointObservationModel<MultipleOf<ObservationModel, SensorCount>>,
               PointSetTransform>>
{
    typedef GaussianFilter<
               StateProcessModel,
               JointParamProcessModel,
               JointObservationModel<MultipleOf<ObservationModel, SensorCount>>,
               PointSetTransform
            > Filter;

    /** \cond INTERNAL */
    /**
     * Represents the factorized model of a set of independent parameters
     * which shall be filtered jointly with the state.
     */
    typedef JointProcessModel<
                MultipleOf<ParamProcessModel, SensorCount>
            > JointParamProcessModel;

    /**
     * Internal joint process model consisting of \c StateProcessModel and
     * the JointProcessModel of multiple ParamProcessModel.
     */
    typedef JointProcessModel<
                StateProcessModel,
                JointParamProcessModel
            > ProcessModel;

    /**
     * Marginal distribution of the state component
     */
    typedef Gaussian<
                typename Traits<StateProcessModel>::State
            > StateMarginalDistribution;

    /**
     * Marginal distribution of the parameter components. The marginal
     * ifself consist of multiple Gaussian marginals, one for each parameter.
     */
    typedef JointDistribution<
                MultipleOf<
                    Gaussian<typename Traits<ParamProcessModel>::State>,
                    SensorCount
                >
            > ParamMarginalDistribution;
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
    typedef typename Traits<StateProcessModel>::Noise StateNoise;
    typedef typename Traits<ParamProcessModel>::Noise ParamNoise;
    typedef typename Traits<ObservationModel>::Noise ObsrvNoise;

    /**
     * Represents the total number of points required by the point set
     * transform.
     */
    enum : signed int
    {
        NumberOfPoints = PointSetTransform::number_of_points(
                             JoinSizes<
                                 State::RowsAtCompileTime,
                                 StateNoise::RowsAtCompileTime,
                                 ObsrvNoise::RowsAtCompileTime
                             >::Size)
    };

    /** \endcond */
};

/**
 * ingroup gaussian_filter_iid
 * \ingroup sigma_point_kalman_filters
 *
 * This \c GaussianFilter represents a factorized implementation of a Sigma
 * Point Kalman Filter. The filter state consists of a coherent state component
 * and a factorized parameter component.
 *
 * \tparam StateProcessModel
 * \tparam ParamProcessModel
 * \tparam SensorCount
 * \tparam ObservationModel
 * \tparam PointSetTransform
 */
template <
    typename StateProcessModel,
    typename ParamProcessModel,
    int SensorCount,
    typename ObservationModel,
    typename PointSetTransform
>
class GaussianFilter<
          StateProcessModel,
          JointProcessModel<MultipleOf<ParamProcessModel, SensorCount>>,
          ObservationModel,
          PointSetTransform
      >
    : /**
       * ProcessModelInterface Base
       */
      public Traits<
          GaussianFilter<
              StateProcessModel,
              JointProcessModel<MultipleOf<ParamProcessModel, SensorCount>>,
              ObservationModel,
              PointSetTransform
          >
      >::ProcessModelBase
{
protected:
    /** \cond INTERNAL */
    typedef GaussianFilter<
                StateProcessModel,
                JointProcessModel<MultipleOf<ParamProcessModel, SensorCount>>,
                ObservationModel,
                PointSetTransform
            > This;

    typedef typename Traits<This>::ProcessModel ProcessModel;
    typedef typename Traits<This>::JointParamProcessModel JointParamProcessModel;
    /** \endcond */

private:
    /**
     * Variates
     */
    enum : char
    {
        a = 0, /**< \brief Coherent state vector component \f$a\f$ */
        v_a,   /**< \brief Noise vector state component \f$a\f$ */

        b,     /**< \brief Joint vector of factorized parameter component $ */
        v_b,   /**< \brief Joint noise vector of factorized parameters */

        b_i,   /**< \brief Single parameter \f$b_i\f$ */
        v_b_i, /**< \brief Noise vector of a singe parameter */

        y,     /**< \brief Joint measurement */
        w,     /**< \brief Joint measurement noise */

        y_i,   /**< \brief Single measurement \f$y_i\f$ of \f$i\f$-th sensor */
        w_i    /**< \brief Noise vector of the \f$i\f$-th sensor */
    };

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
     * \param state_process_model
     * \param parameter_process_model
     * \param obsrv_model
     * \param point_set_transform
     * \param parameter_count
     */
    GaussianFilter(
            const std::shared_ptr<StateProcessModel>& state_process_model,
            const std::shared_ptr<ParamProcessModel>& parameter_process_model,
            const std::shared_ptr<ObservationModel> obsrv_model,
            const std::shared_ptr<PointSetTransform>& point_set_transform,
            int parameter_count = ToDimension<SensorCount>::Value)
        : state_process_model_(state_process_model),
          parameter_process_model_(parameter_process_model),
          joint_param_process_model_(
              std::make_shared<JointParamProcessModel>(
                  parameter_process_model_,
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



        point_set_transform_->forward(prior_dist, );
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
        case a:     return state_process_model_->state_dimension();
        case v_a:   return obsrv_model_->noi_dimension();

        case b:     return joint_param_process_model_->state_dimension();
        case v_b:   return joint_param_process_model_->noise_dimension();

        case b_i:   return parameter_process_model_->state_dimension();;
        case v_b_i: return parameter_process_model_->noise_dimension();

        case y:     return obsrv_model_->observation_dimension();
        case w:     return obsrv_model_->noise_dimension();

        case y_i:   return obsrv_model_->observation_dimension()/SensorCount;
        case w_i:   return obsrv_model_->noise_dimension()/SensorCount;

        default:
            // throw!
            break;
        }
    }

private:
    std::shared_ptr<StateProcessModel> state_process_model_;
    std::shared_ptr<ParamProcessModel> parameter_process_model_;
    std::shared_ptr<JointParamProcessModel> joint_param_process_model_;

    std::shared_ptr<ProcessModel> process_model_;
    std::shared_ptr<ObservationModel> obsrv_model_;
    std::shared_ptr<PointSetTransform> point_set_transform_;
};

}

#endif

const int a_noise_dim = state_process_model_->state_dimension();
