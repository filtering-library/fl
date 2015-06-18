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
 * \file gaussian_filter_ukf.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__FILTER__GAUSSIAN__GAUSSIAN_FILTER_UKF_HPP
#define FL__FILTER__GAUSSIAN__GAUSSIAN_FILTER_UKF_HPP

#include <map>
#include <tuple>
#include <memory>

#include <fl/util/meta.hpp>
#include <fl/util/traits.hpp>
#include <fl/util/profiling.hpp>

#include <fl/exception/exception.hpp>
#include <fl/filter/filter_interface.hpp>
#include <fl/filter/gaussian/point_set.hpp>
#include <fl/filter/gaussian/feature_policy.hpp>

#include <fl/model/observation/joint_observation_model.hpp>

namespace fl
{

template <typename...> class GaussianFilter;

/**
 * GaussianFilter Traits
 */
template <
    typename ProcessModel,
    typename ObservationModel,
    typename PointSetTransform,
    template <typename...T> class FeaturePolicy
>
struct Traits<
           GaussianFilter<
               ProcessModel,
               ObservationModel,
               PointSetTransform,
               FeaturePolicy<>>>
{
    /*
     * Required concept (interface) types
     *
     * - Ptr
     * - State
     * - Input
     * - Observation
     * - Belief
     */
    typedef typename Traits<ProcessModel>::State State;
    typedef typename Traits<ProcessModel>::Input Input;
    typedef typename Traits<ObservationModel>::Obsrv Obsrv;

    /**
     * Represents the underlying distribution of the estimated state. In
     * case of a Point Based Kalman filter, the distribution is a simple
     * Gaussian with the dimension of the \c State.
     */
    typedef Gaussian<State> Belief;

    /** \cond INTERNAL */
    typedef typename Traits<ProcessModel>::Noise StateNoise;
    typedef typename Traits<ObservationModel>::Noise ObsrvNoise;


    typedef FeaturePolicy<Obsrv> FeatureMapping;
    typedef typename Traits<FeatureMapping>::ObsrvFeature ObsrvFeature;

    /**
     * Represents the total number of points required by the point set
     * transform.
     *
     * The number of points is a function of the joint Gaussian of which we
     * compute the transform. In this case the joint Gaussian consists of
     * the Gaussian of the state, the state noise Gaussian and the
     * observation noise Gaussian. The latter two are needed since we assume
     * models with non-additive noise.
     *
     * The resulting number of points determined by the employed transform
     * passed via \c PointSetTransform. If on or more of the marginal
     * Gaussian sizes is dynamic, the number of points is dynamic as well.
     * That is, the number of points cannot be known at compile time and
     * will be allocated dynamically on run time.
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

    typedef PointSet<State, NumberOfPoints> StatePointSet;
    typedef PointSet<Obsrv, NumberOfPoints> ObsrvPointSet;
    typedef PointSet<StateNoise, NumberOfPoints> StateNoisePointSet;
    typedef PointSet<ObsrvNoise, NumberOfPoints> ObsrvNoisePointSet;
    typedef PointSet<ObsrvFeature, NumberOfPoints> ObsrvFeaturePointSet;
    /** \endcond */
};

/**
 * \ingroup sigma_point_kalman_filters
 * \ingroup unscented_kalman_filter
 *
 * GaussianFilter represents all filters based on Gaussian distributed systems.
 * This includes the Kalman Filter and filters using non-linear models such as
 * Sigma Point Kalman Filter family.
 *
 * \tparam ProcessModel
 * \tparam ObservationModel
 *
 */
template<
    typename ProcessModel,
    typename ObservationModel,
    typename PointSetTransform,
    template <typename...T> class FeaturePolicy
>
class GaussianFilter<
          ProcessModel,
          ObservationModel,
          PointSetTransform,
          FeaturePolicy<>>
    :
    /* Implement the filter interface */
    public FilterInterface<
              GaussianFilter<
                  ProcessModel,
                  ObservationModel,
                  PointSetTransform,
                  FeaturePolicy<>>>
{
private:
    /** Typdef of \c This for #from_traits(TypeName) helper */
    typedef GaussianFilter<
                ProcessModel,
                ObservationModel,
                PointSetTransform,
                FeaturePolicy<>
            > This;

    typedef from_traits(StateNoise);
    typedef from_traits(ObsrvNoise);
    typedef from_traits(ObsrvFeature);
    typedef from_traits(StatePointSet);
    typedef from_traits(ObsrvPointSet);
    typedef from_traits(StateNoisePointSet);
    typedef from_traits(ObsrvNoisePointSet);
    typedef from_traits(ObsrvFeaturePointSet);

public:
    typedef from_traits(State);
    typedef from_traits(Input);
    typedef from_traits(Obsrv);
    typedef from_traits(Belief);
    typedef from_traits(FeatureMapping);

public:
    /**
     * Creates a Gaussian filter
     *
     * \param process_model         Process model instance
     * \param obsrv_model           Obsrv model instance
     * \param point_set_transform   Point set tranfrom such as the unscented
     *                              transform
     */
    GaussianFilter(const ProcessModel& process_model,
                   const ObservationModel& obsrv_model,
                   const PointSetTransform& point_set_transform,
                   const FeatureMapping& feature_mapping = FeatureMapping())
        : process_model_(process_model),
          obsrv_model_(obsrv_model),
          point_set_transform_(point_set_transform),
          feature_mapping_(feature_mapping),
          /*
           * Set the augmented Gaussian dimension.
           *
           * The global dimension is dimension of the augmented Gaussian which
           * consists of state Gaussian, state noise Gaussian and the
           * observation noise Gaussian.
           */
          global_dimension_(process_model_.state_dimension()
                            + process_model_.noise_dimension()
                            + obsrv_model_.noise_dimension()),
          /*
           * Initialize the points-set Gaussian (e.g. sigma points) of the
           * \em state noise. The number of points is determined by the
           * augmented Gaussian with the dimension  global_dimension_
           */
          X_Q(process_model_.noise_dimension(),
              PointSetTransform::number_of_points(global_dimension_)),
          /*
           * Initialize the points-set Gaussian (e.g. sigma points) of the
           * \em observation noise. The number of points is determined by the
           * augmented Gaussian with the dimension global_dimension_
           */
          X_R(obsrv_model_.noise_dimension(),
              PointSetTransform::number_of_points(global_dimension_))
    {
        /*
         * pre-compute the state noise points from the standard Gaussian
         * distribution with the dimension of the state noise and store the
         * points in the X_Q PointSet
         *
         * The points are computet for the marginal Q of the global Gaussian
         * with the dimension global_dimension_ as depecited below
         *
         *    [ P  0  0 ]
         * -> [ 0  Q  0 ] -> [X_Q[1]  X_Q[2] ... X_Q[p]]
         *    [ 0  0  R ]
         *
         * p is the number of points determined be the transform type
         *
         * The transform takes the global dimension (dim(P) + dim(Q) + dim(R))
         * and the dimension offset dim(P) as parameters.
         */
        point_set_transform_.forward(
            Gaussian<StateNoise>(process_model_.noise_dimension()),
            global_dimension_,
            process_model_.state_dimension(),
            X_Q);

        /*
         * pre-compute the observation noise points from the standard Gaussian
         * distribution with the dimension of the observation noise and store
         * the points in the X_R PointSet
         *
         * The points are computet for the marginal R of the global Gaussian
         * with the dimension global_dimension_ as depecited below
         *
         *    [ P  0  0 ]
         *    [ 0  Q  0 ]
         * -> [ 0  0  R ] -> [X_R[1]  X_R[2] ... X_R[p]]
         *
         * again p is the number of points determined be the transform type
         *
         * The transform takes the global dimension (dim(P) + dim(Q) + dim(R))
         * and the dimension offset dim(P) + dim(Q) as parameters.
         */
        point_set_transform_.forward(
            Gaussian<ObsrvNoise>(obsrv_model_.noise_dimension()),
            global_dimension_,
            process_model_.state_dimension()
            + process_model_.noise_dimension(),
            X_R);

        /*
         * Setup the point set of the observation predictions
         */
        const size_t point_count =
                PointSetTransform::number_of_points(global_dimension_);

        X_y.resize(point_count);
        X_y.dimension(obsrv_model_.obsrv_dimension());

        X_fy.resize(point_count);
        X_fy.dimension(
            feature_mapping_.feature_dimension(obsrv_model_.obsrv_dimension()));

        /*
         * Setup the point set of the state predictions
         */
        X_r.resize(point_count);
        X_r.dimension(process_model_.state_dimension());
    }

    /**
     * \copydoc FilterInterface::predict
     */
    virtual void predict(double delta_time,
                         const Input& input,
                         const Belief& prior_belief,
                         Belief& predicted_belief)
    {
        {
            point_set_transform_.forward(
                Gaussian<StateNoise>(process_model_.noise_dimension()),
                global_dimension_,
                process_model_.state_dimension(),
                X_Q);
        }

        /*
         * Compute the state points from the given prior state Gaussian
         * distribution and store the points in the X_r PointSet
         *
         * The points are computet for the marginal R of the global Gaussian
         * with the dimension global_dimension_ as depecited below
         *
         * -> [ P  0  0 ] -> [X_r[1]  X_r[2] ... X_r[p]]
         *    [ 0  Q  0 ]
         *    [ 0  0  R ]
         *
         * The transform takes the global dimension (dim(P) + dim(Q) + dim(R))
         * and the dimension offset 0 as parameters.
         */
        point_set_transform_.forward(prior_belief,
                                      global_dimension_,
                                      0,
                                      X_r);

        /*
         * Predict each point X_r[i] and store the prediction back in X_r[i]
         *
         * X_r[i] = f(X_r[i], X_Q[i], u)
         */
        const size_t point_count = X_r.count_points();
        for (size_t i = 0; i < point_count; ++i)
        {
            X_r[i] = process_model_.predict_state(delta_time,
                                                  X_r[i],
                                                  X_Q[i],
                                                  input);
        }

        /*
         * Obtain the centered points matrix of the prediction. The columns of
         * this matrix are the predicted points with zero mean. That is, the
         * sum of the columns in P is zero.
         *
         * P = [X_r[1]-mu_r  X_r[2]-mu_r  ... X_r[n]-mu_r]
         *
         * with weighted mean
         *
         * mu_r = Sum w_mean[i] X_r[i]
         */
//        auto&& X = X_r.centered_points();

        /*
         * Obtain the weights of point as a vector
         *
         * W = [w_cov[1]  w_cov[2]  ... w_cov[n]]
         *
         * Note that the covariance weights are used.
         */
//        auto&& W = X_r.covariance_weights_vector();

        /*
         * Compute and set the moments
         *
         * The first moment is simply the weighted mean of points.
         * The second centered moment is determined by
         *
         * C = Sum W[i,i] * (X_r[i]-mu_r)(X_r[i]-mu_r)^T
         *   = P * W * P^T
         *
         * given that W is the diagonal matrix
         */
//        predicted_belief.mean(X_r.mean());
//        predicted_belief.covariance(X * W.asDiagonal() * X.transpose());
    }

    /**
     * \copydoc FilterInterface::update
     */
    virtual void update(const Obsrv& obsrv,
                        const Belief& predicted_belief,
                        Belief& posterior_belief)
    {
//        point_set_transform_.forward(predicted_belief,
//                                     global_dimension_,
//                                     0,
//                                     X_r);

        {
            point_set_transform_.forward(
                Gaussian<ObsrvNoise>(obsrv_model_.noise_dimension()),
                global_dimension_,
                process_model_.state_dimension()
                + process_model_.noise_dimension(),
                X_R);
        }

        const size_t point_count = X_r.count_points();

        for (size_t i = 0; i < point_count; ++i)
        {
            X_y[i] = obsrv_model_.predict_obsrv(X_r[i],
                                                X_R[i],
                                                1.0 /* delta time */);
        }

        auto obsrv_prediction = X_y.mean();
        auto centered_prediction = X_y.centered_points();
        auto var = (centered_prediction.array().pow(2).rowwise().sum() / double(point_count)).eval();
        for (size_t i = 0; i < point_count; ++i)
        {
            X_fy[i] = feature_mapping_.extract(X_y[i], obsrv_prediction, var);
        }

        ObsrvFeature y = feature_mapping_.extract(obsrv, obsrv_prediction, var);

        auto&& prediction = X_fy.center();
        auto&& Y = X_fy.points();
        auto&& W = X_r.covariance_weights_vector();
        auto&& X = X_r.centered_points();

        auto innovation = (y - prediction).eval();
        auto cov_xx = (X * W.asDiagonal() * X.transpose()).eval();
        auto cov_yy = (Y * W.asDiagonal() * Y.transpose()).eval();
        auto cov_xy = (X * W.asDiagonal() * Y.transpose()).eval();
        auto K = (cov_xy * cov_yy.inverse()).eval();

        posterior_belief.mean(X_r.mean() + K * innovation);
        posterior_belief.covariance(cov_xx - K * cov_yy * K.transpose());
    }

    /**
     * \copydoc FilterInterface::predict_and_update
     */
    virtual void predict_and_update(double delta_time,
                                    const Input& input,
                                    const Obsrv& observation,
                                    const Belief& prior_belief,
                                    Belief& posterior_belief)
    {
        predict(delta_time, input, prior_belief, posterior_belief);
        update(observation, posterior_belief, posterior_belief);
    }

    ProcessModel& process_model() { return process_model_; }
    ObservationModel& obsrv_model() { return obsrv_model_; }
    PointSetTransform& point_set_transform() { return point_set_transform_; }
    FeatureMapping& feature_mapping() { return feature_mapping_; }

    const ProcessModel& process_model() const
    {
        return process_model_;
    }

    const ObservationModel& obsrv_model() const
    {
        return obsrv_model_;
    }

    const PointSetTransform& point_set_transform() const
    {
        return point_set_transform_;
    }

    const FeatureMapping& feature_mapping() const
    {
        return feature_mapping_;
    }

    virtual Belief create_state_distribution() const
    {
        auto state_distr = Belief(process_model().state_dimension());

        return state_distr;
    }

public:
    double threshold;
    double inv_sigma;
    bool print_details;

public:
    ProcessModel process_model_;
    ObservationModel obsrv_model_;
    PointSetTransform point_set_transform_;
    FeatureMapping feature_mapping_;

    /** \cond INTERNAL */
    /**
     * \brief The global dimension is dimension of the augmented Gaussian which
     * consists of state Gaussian, state noise Gaussian and the
     * observation noise Gaussian.
     */
    const size_t global_dimension_;

    /**
     * \brief Represents the point-set of the state
     */
    StatePointSet X_r;

    /**
     * \brief Represents the point-set of the observation
     */
    ObsrvPointSet X_y;

    /**
     * \brief Represents the point-set of the feature of observations
     */
    ObsrvFeaturePointSet X_fy;

    /**
     * \brief Represents the points-set Gaussian (e.g. sigma points) of the
     * \em state noise. The number of points is determined by the augmented
     * Gaussian with the dimension #global_dimension_
     */
    StateNoisePointSet X_Q;

    /**
     * \brief Represents the points-set Gaussian (e.g. sigma points) of the
     * \em observation noise. The number of points is determined by the
     * augmented Gaussian with the dimension #global_dimension_
     */
    ObsrvNoisePointSet X_R;
    /** \endcond */

public:
    /** \cond INTERNAL */
    /* Dungeon - keep put! */
    decltype(X_y.mean()) prediction;
    decltype(prediction) innovation;
    /** \endcond */
};

#ifdef TEMPLATE_ARGUMENTS
    #undef TEMPLATE_ARGUMENTS
#endif


#define TEMPLATE_ARGUMENTS \
    JointProcessModel< \
       ProcessModel, \
       JointProcessModel<MultipleOf<LocalParamModel, Count>>>, \
    Adaptive<JointObservationModel<MultipleOf<LocalObsrvModel, Count>>>, \
    PointSetTransform,\
    FeaturePolicy<>

#ifndef GENERATING_DOCUMENTATION
template <
    typename ProcessModel,
    typename LocalObsrvModel,
    typename LocalParamModel,
    int Count,
    typename PointSetTransform,
    template <typename...T> class FeaturePolicy
>
#endif
struct Traits<
           GaussianFilter<
               ProcessModel,
               Join<MultipleOf<Adaptive<LocalObsrvModel, LocalParamModel>, Count>>,
               PointSetTransform,
               FeaturePolicy<>,
               Options<NoOptions>
            >
        >
    : Traits<GaussianFilter<TEMPLATE_ARGUMENTS>>
{ };

#ifndef GENERATING_DOCUMENTATION
template <
    typename ProcessModel,
    typename LocalObsrvModel,
    typename LocalParamModel,
    int Count,
    typename PointSetTransform,
    template <typename...T> class FeaturePolicy
>
#endif
class GaussianFilter<
          ProcessModel,
          Join<MultipleOf<Adaptive<LocalObsrvModel, LocalParamModel>, Count>>,
          PointSetTransform,
          FeaturePolicy<>,
          Options<NoOptions>
      >
    : public GaussianFilter<TEMPLATE_ARGUMENTS>
{
public:
    typedef GaussianFilter<TEMPLATE_ARGUMENTS> Base;

    GaussianFilter(
        const ProcessModel& state_process_model,
        const LocalParamModel& param_process_model,
        const LocalObsrvModel& obsrv_model,
        const PointSetTransform& point_set_transform,
        const typename Traits<Base>::FeatureMapping& feature_mapping
            = typename Traits<Base>::FeatureMapping(),
        int parameter_count = Count)
            : Base(
                { state_process_model, {param_process_model, parameter_count} },
                { obsrv_model, parameter_count },
                point_set_transform,
                feature_mapping)
    { }
};

#undef TEMPLATE_ARGUMENTS

}

#endif
