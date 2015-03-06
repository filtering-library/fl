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
#include <fl/util/profiling.hpp>

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

/**
 * \ingroup gaussian_filter_iid
 * Traits of Factorized GaussianFilter for IID Parameters
 */
#ifndef GENERATING_DOCUMENTATION
template <
    int Count,
    typename StateProcessModel,
    typename LocalParamModel,
    typename LocalObsrvModel,
    typename PointSetTransform
>
#endif
struct Traits<
           GaussianFilter<
                JointProcessModel<
                    StateProcessModel,
                    JointProcessModel<MultipleOf<LocalParamModel, Count>>>,
                    Adaptive<
                        JointObservationModel<
                            MultipleOf<LocalObsrvModel, Count>
                        >
                    >,
                PointSetTransform,
                Options<FactorizeParams>>>
{
    /** \cond INTERNAL ****************************************************** */
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
    typedef Adaptive<
                JointObservationModel<
                    MultipleOf<LocalObsrvModel, Count>
                >
            > ObservationModel;

    /**
     * \brief Marginal distribution of the state component
     */
    typedef Gaussian<
                typename Traits<StateProcessModel>::State
            > LocalStateDistr;

    /**
     * \brief Marginal distribution of the parameter components. The marginal
     * ifself consist of multiple Gaussian marginals, one for each parameter.
     */
    typedef JointDistribution<
                MultipleOf<
                    Gaussian<typename Traits<LocalParamModel>::State>,
                    Count>
            > ParamMarginalDistr;
    /** \endcond ************************************************************ */

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
                LocalStateDistr,
                ParamMarginalDistr
            > StateDistribution;

    /** \cond INTERNAL */    
    typedef typename Traits<ProcessModel>::Noise StateNoise;
    typedef typename Traits<ObservationModel>::Noise ObsrvNoise;

    typedef typename Traits<StateProcessModel>::State LocalState;
    typedef typename Traits<StateProcessModel>::Noise LocalStateNoise;
    typedef typename Traits<LocalParamModel>::State LocalParam;
    typedef typename Traits<LocalParamModel>::Noise LocalParamNoise;
    typedef typename Traits<LocalObsrvModel>::Obsrv LocalObsrv;
    typedef typename Traits<LocalObsrvModel>::Noise LocalObsrvNoise;

    /**
     * \brief Represents the total number of points required by the point set
     * transform.
     */
    enum : signed int
    {
        NumberOfPoints = PointSetTransform::number_of_points(
                             JoinSizes<
                                 LocalState::RowsAtCompileTime,
                                 LocalStateNoise::RowsAtCompileTime,
                                 LocalParam::RowsAtCompileTime,
                                 LocalParamNoise::RowsAtCompileTime,
                                 LocalObsrvNoise::RowsAtCompileTime
                             >::Size)
    };

    typedef PointSet<State, NumberOfPoints> StatePointSet;
    typedef PointSet<Obsrv, NumberOfPoints> ObsrvPointSet;
    typedef PointSet<StateNoise, NumberOfPoints> StateNoisePointSet;
    typedef PointSet<ObsrvNoise, NumberOfPoints> ObsrvNoisePointSet;


    typedef PointSet<LocalState, NumberOfPoints> LocalStatePointSet;
    typedef PointSet<LocalParam, NumberOfPoints> LocalParamPointSet;
    typedef PointSet<LocalObsrv, NumberOfPoints> LocalObsrvPointSet;
    typedef PointSet<LocalStateNoise, NumberOfPoints> LocalStateNoisePointSet;
    typedef PointSet<LocalParamNoise, NumberOfPoints> LocalParamNoisePointSet;
    typedef PointSet<LocalObsrvNoise, NumberOfPoints> LocalObsrvNoisePointSet;

    typedef Eigen::Array<LocalParamPointSet, Count, 1> LocalParamPointSets;

    typedef Gaussian<LocalParam> LocalParamDistr;
    typedef Gaussian<LocalObsrv> LocalObsrvDistr;
    typedef Gaussian<LocalStateNoise> LocalStateNoiseDistr;
    typedef Gaussian<LocalParamNoise> LocalParamNoiseDistr;
    typedef Gaussian<LocalObsrvNoise> LocalObsrvNoiseDistr;

    typedef typename Traits<LocalStateDistr>::SecondMoment LocalStateCov;
    typedef typename Traits<LocalParamDistr>::SecondMoment LocalParamCov;
    typedef typename Traits<LocalObsrvDistr>::SecondMoment LocalObsrvCov;
    typedef typename Traits<LocalStateNoiseDistr>::SecondMoment LocalStateNoiseCov;
    typedef typename Traits<LocalParamNoiseDistr>::SecondMoment LocalParamNoiseCov;
    typedef typename Traits<LocalObsrvNoiseDistr>::SecondMoment LocalObsrvNoiseCov;
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
    int Count,
    typename StateProcessModel,
    typename LocalParamModel,
    typename LocalObsrvModel,
    typename PointSetTransform
>
#endif
class GaussianFilter<
          JointProcessModel<
              StateProcessModel,
              JointProcessModel<MultipleOf<LocalParamModel, Count>>>,
              Adaptive<
                  JointObservationModel<
                      MultipleOf<LocalObsrvModel, Count>
                  >
              >,
          PointSetTransform,
          Options<FactorizeParams>>
    :
    /* Implement the filter interface */
    public FilterInterface<
               GaussianFilter<
                    JointProcessModel<
                        StateProcessModel,
                        JointProcessModel<MultipleOf<LocalParamModel, Count>>>,
                    Adaptive<
                        JointObservationModel<
                            MultipleOf<LocalObsrvModel, Count>
                        >
                    >,
                    PointSetTransform,
                    Options<FactorizeParams>>>
{
protected:
    /** \cond INTERNAL ****************************************************** */
    typedef GaussianFilter This;

    /* Models */
    typedef from_traits(ProcessModel);
    typedef from_traits(ObservationModel);
    typedef from_traits(JointParamProcessModel);

    /* Sigma Point Sets */
    typedef from_traits(StatePointSet);
    typedef from_traits(ObsrvPointSet);
    typedef from_traits(StateNoisePointSet);
    typedef from_traits(ObsrvNoisePointSet);

    typedef from_traits(LocalStatePointSet);
    typedef from_traits(LocalParamPointSet);
    typedef from_traits(LocalObsrvPointSet);

    typedef from_traits(LocalParamPointSets);

    typedef from_traits(LocalStateNoisePointSet);
    typedef from_traits(LocalParamNoisePointSet);
    typedef from_traits(LocalObsrvNoisePointSet);

    /* Distributions */
//    typedef from_traits(LocalStateDistr);
//    typedef from_traits(LocalParamDistr);
//    typedef from_traits(LocalObsrvDistr);
//    typedef from_traits(LocalStateNoiseDistr);
//    typedef from_traits(LocalParamNoiseDistr);
//    typedef from_traits(LocalObsrvNoiseDistr);

    /* Variates */
    typedef from_traits(LocalState);
    typedef from_traits(LocalStateNoise);
    typedef from_traits(LocalParam);
    typedef from_traits(LocalParamNoise);
    typedef from_traits(LocalObsrv);
    typedef from_traits(LocalObsrvNoise);

    /* Second moments */
//    typedef from_traits(LocalStateCov);
//    typedef from_traits(LocalParamCov);
//    typedef from_traits(LocalObsrvCov);
//    typedef from_traits(LocalStateNoiseCov);
//    typedef from_traits(LocalParamNoiseCov);
//    typedef from_traits(LocalObsrvNoiseCov);

    /**
     * \enum Variates
     */
    enum Variate
    {
        a = 0, /**< Coherent state vector component \f$a\f$ */
        b = 1, /**< Joint vector of factorized parameter component $ */
        v_a,   /**< Noise vector state component \f$a\f$ */
        v_b,   /**< Joint noise vector of factorized parameters */
        v,     /**< Joint State noise */
        b_i,   /**< Single parameter \f$b_i\f$ */
        v_b_i, /**< Noise vector of a singe parameter */
        y,     /**< Joint measurement */
        w,     /**< Joint measurement noise */
        y_i,   /**< Single measurement \f$y_i\f$ of \f$i\f$-th sensor */
        w_i,   /**< Noise vector of the \f$i\f$-th sensor */
        x,     /**< Dimension of the entire state [a b_1 b_2 ... b_n]*/
        u,
        u_a,
        u_b_i
    };
    /** \endcond ************************************************************ */

public:
    /* public concept interface types */
    typedef from_traits(State);
    typedef from_traits(Input);
    typedef from_traits(Obsrv);
    typedef from_traits(StateNoise);
    typedef from_traits(ObsrvNoise);
    typedef from_traits(StateDistribution);

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
            int param_count = Count)
        : f_a_(state_process_model),
          f_b_i(param_process_model),
          f_b_(f_b_i, param_count),
          f_(f_a_, f_b_),
          h_i_(obsrv_model),
          h_(h_i_, param_count),
          transform_(point_set_transform),
          param_count_(param_count),

          /*
           * Determine the marginal distribution dimension of
           * [a,  v_a,  b_i,  v_b_i,  w_i]
           */
          dim_marginal_(dim(a) + dim(v_a) + dim(b_i) + dim(v_b_i) + dim(w_i)),

          /*
           * Determine the number of points given the join marginal of
           *
           * [a,  v_a,  b_i,  v_b_i,  w_i]
           */
          point_count_(point_set_transform.number_of_points(dim_marginal_))
    {
        assert(param_count > 0);

        X_.resize(dim(x), point_count_);
        Y_.resize(dim(y), point_count_);
        X_v_.resize(dim(v), point_count_);
        X_w_.resize(dim(w), point_count_);

        X_a_.resize(dim(a), point_count_);
        X_b_.resize(param_count, 1);
        for (int i = 0; i < param_count_; ++i)
        {
            X_b_(i).resize(dim(b_i), point_count_);
        }

        /* Create noise standard gaussian distributions */
        auto N_a = Gaussian<LocalStateNoise>(dim(v_a));
        auto N_b_i = Gaussian<LocalParamNoise>(dim(v_b_i));
        auto N_y_i = Gaussian<LocalObsrvNoise>(dim(w_i));

        /* Pre-Compute the sigma points of noise distributions */
        auto X_v_a = LocalStateNoisePointSet(dim(v_a), point_count_);
        auto X_v_b_i = LocalParamNoisePointSet(dim(v_b_i), point_count_);
        auto X_w_y_i = LocalObsrvNoisePointSet(dim(w_i), point_count_);

        int dim_offset = dim(a);
        transform_.forward(N_a, dim_marginal_, dim_offset, X_v_a);

        dim_offset += dim(v_a);
        transform_.forward(N_b_i, dim_marginal_, dim_offset, X_v_b_i);

        dim_offset += dim(v_b_i);
        transform_.forward(N_y_i, dim_marginal_, dim_offset, X_w_y_i);

        /* Create X_v and X_w joint sigma points */
        X_v_.points().topRows(dim(v_a)) = X_v_a.points();
        for (int i = 0; i < param_count; ++i)
        {
            X_v_.points().middleRows(dim(v_a) + i * dim(v_b_i), dim(v_b_i)) =
                X_v_b_i.points();

            X_w_.points().middleRows(i * dim(w_i), dim(w_i)) =
                X_w_y_i.points();
        }

        X_v_a_ = X_v_a;
        X_v_b_i_ = X_v_b_i;
        X_w_y_i_ = X_w_y_i;
    }

    LocalStateNoisePointSet X_v_a_;
    LocalParamNoisePointSet X_v_b_i_;
    LocalObsrvNoisePointSet X_w_y_i_;

    /**
     * \copydoc FilterInterface::predict
     */
    virtual void predict(double dt,
                         const Input& u,
                         const StateDistribution& prior_dist,
                         StateDistribution& predicted_dist)
    {
        /**
         * Compute sigma points of part X_a and each of X_b(i)
         */
        transform(prior_dist, X_a_, X_b_);

        combine(X_a_, X_b_, X_);

        /* Predict all sigma points */        
        for (int i = 0; i < point_count_; ++i)
        {
            X_[i] = f_(dt, X_[i], X_v_[i], u);
        }

//        for (int i = 0; i < point_count_; ++i)
//        {
//            X_a.points().col(i) = f_a_.predict_state(
//                                      delta_time,
//                                      X_a.points().col(i),
//                                      X_v_a_.points().col(i),
//                                      input.topRows(dim(u_a)));

//            for (int k = 0; k < param_count_; ++k)
//            {
//                X_b(k).points().col(i) = f_b_i.predict_state(
//                                             delta_time,
//                                             X_b(k).points().col(i),
//                                             X_v_b_i_.points().col(i),
//                                             input.middleRows(
//                                                 dim(u_a) + k * dim(u_b_i),
//                                                 dim(u_b_i)));
//            }

//            // alternatively combine X_a and X_b(i) into X and predict all
//            // X_pred(i) = f_(X(i))
//        }
    }    

    /**
     * \copydoc FilterInterface::update
     */
    virtual void update(const Obsrv& y,
                        const StateDistribution& predicted_dist,
                        StateDistribution& posterior_dist)
    {
        for (int i = 0; i < param_count_; ++i)
        {
            h_i_.id(i);
        }
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

    ProcessModel& process_model() { return f_; }
    ObservationModel& observation_model() { return h_; }
    PointSetTransform& point_set_transform() { return transform_; }

    StateProcessModel& state_process_model() { return f_a_; }
    JointParamProcessModel& joint_param_process_model() { return f_b_; }

public:
    /* ** Helpers *********************************************************** */

    /**
     * \return Dimension of the specified variate
     *
     * \throws Exception("Unknown variate component.")
     */
    const int dim(Variate variate) const
    {
        switch (variate)
        {
        case a:     return f_a_.state_dimension();
        case v_a:   return f_a_.noise_dimension();
        case b:     return f_b_.state_dimension();
        case v_b:   return f_b_.noise_dimension();
        case b_i:   return f_b_i.state_dimension();
        case v_b_i: return f_b_i.noise_dimension();
        case v:     return f_.noise_dimension();
        case y:     return h_.obsrv_dimension();
        case w:     return h_.noise_dimension();
        case y_i:   return h_i_.obsrv_dimension();
        case w_i:   return h_i_.noise_dimension();
        case x:     return f_.state_dimension();
        case u:     return f_.input_dimension();
        case u_a:   return f_a_.input_dimension();
        case u_b_i: return f_b_i.input_dimension();

        default:
            // throw!
            throw Exception("Unknown variate component.");
        }
    }

    /**
     * Compute sigma points of part X_a and each of X_b(i) for a given
     * state distribution \c distr
     */
    template <typename Xa, typename Xb>
    void transform(const StateDistribution& distr,
                   Xa& x_a, Xb& x_b)
    {
        // transform X_a part
        auto&& prior_a = std::get<a>(distr.distributions());
        transform_.forward(prior_a, dim_marginal_, 0, x_a);

        // transform all X_b(i)
        auto&& prior_b = std::get<b>(distr.distributions());
        const int offset = dim(a) + dim(v_a);
        for (int i = 0; i < param_count_; ++i)
        {
            transform_.forward(
                prior_b.distribution(i), dim_marginal_, offset, x_b(i));
        }
    }

    template <typename Xa, typename Xb, typename X_>
    void combine(const Xa& x_a, const Xb& x_b, X_& x)
    {
        x.points().topRows(dim(a)) = x_a.points();
        for (int i = 0; i < param_count_; ++i)
        {
            x.points().middleRows(dim(a) + i * dim(b_i), dim(b_i)) =
                x_b(i).points();
        }
    }

private:
    /* Model */
    StateProcessModel f_a_;
    LocalParamModel f_b_i;
    JointParamProcessModel f_b_;
    ProcessModel f_;

    LocalObsrvModel h_i_;
    ObservationModel h_;

    PointSetTransform transform_;

protected:
    /* Data */
    const int param_count_;
    const int dim_marginal_;
    const int point_count_;

    LocalStatePointSet X_a_;
    LocalParamPointSets X_b_;

    StateNoisePointSet X_v_;
    ObsrvNoisePointSet X_w_;
    StatePointSet X_;
    ObsrvPointSet Y_;
};


/**
 * \ingroup gaussian_filter_iid
 *
 * Translates the filter specification of the form
 *
 *   GaussianFilter< Process, Join<Many<Adaptive<Sensor, ParamModel>>> >
 *
 * to the form taking a joint process and a joint observation model
 *
 *   GaussianFilter< JointProcess, JointObsrv >
 */

# ifndef GENERATING_DOCUMENTATION
template <
    int Count,
    typename StateProcessModel,
    typename ObsrvModel,
    typename ParamProcess,
    typename PointTransform
>
#endif
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
                Adaptive<JointObservationModel<MultipleOf<ObsrvModel, Count>>>,
                PointTransform,
                Options<FactorizeParams>>
{
public:
    typedef GaussianFilter<
                JointProcessModel<
                    StateProcessModel,
                    JointProcessModel<MultipleOf<ParamProcess, Count>>
                >,
                Adaptive<JointObservationModel<MultipleOf<ObsrvModel, Count>>>,
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
