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

#include <boost/shared_ptr.hpp>

#include <fl/util/meta.hpp>
#include <fl/util/traits.hpp>

#include <fl/exception/exception.hpp>
#include <fl/filter/filter_interface.hpp>
#include <fl/filter/gaussian/point_set.hpp>

namespace fl
{

// Forward declarations
template <typename...> class GaussianFilter;

/**
 * GaussianFilter Traits
 */
template <typename ProcessModel,
          typename ObsrvModel,
          typename PointSetTransform>
struct Traits<GaussianFilter<ProcessModel, ObsrvModel, PointSetTransform>>
{
    typedef GaussianFilter<ProcessModel, ObsrvModel, PointSetTransform> Filter;

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
    typedef typename Traits<ObsrvModel>::Observation Observation;

    /**
     * Represents the underlying distribution of the estimated state. In the
     * case of a Point Based Kalman filter, the distribution is a simple
     * Gaussian with the dimension of the \c State.
     */
    typedef fl::Gaussian<State> StateDistribution;

    /** \cond INTERNAL */
    typedef typename Traits<ProcessModel>::Noise StateNoise;
    typedef typename Traits<ObsrvModel>::Noise ObsrvNoise;

    enum
    {
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
        NumberOfPoints = PointSetTransform::number_of_points(
                            JoinSizes<
                                State::RowsAtCompileTime,
                                StateNoise::RowsAtCompileTime,
                                ObsrvNoise::RowsAtCompileTime
                            >::Size)
    };

    typedef PointSet<State, NumberOfPoints> StatePointSet;
    typedef PointSet<Observation, NumberOfPoints> ObsrvPointSet;
    typedef PointSet<StateNoise, NumberOfPoints> StateNoisePointSet;
    typedef PointSet<ObsrvNoise, NumberOfPoints> ObsrvNoisePointSet;

    /**
     * \brief KalmanGain Matrix
     */
    typedef Eigen::Matrix<
                typename StateDistribution::Scalar,
                State::RowsAtCompileTime,
                Observation::RowsAtCompileTime
            > KalmanGain;
    /** \endcond */
};

/**
 * GaussianFilter represents all filters based on Gaussian distributed systems.
 * This includes the Kalman Filter and filters using non-linear models such as
 * Sigma Point Kalman Filter family.
 *
 * \tparam ProcessModel
 * \tparam ObsrvModel
 *
 * \ingroup filters
 */
template<
    typename ProcessModel,
    typename ObsrvModel,
    typename PointSetTransform
>
class GaussianFilter<ProcessModel, ObsrvModel, PointSetTransform>
    :
    /* Implement the conceptual filter interface */
    public FilterInterface<
               GaussianFilter<ProcessModel, ObsrvModel, PointSetTransform>>

{
protected:
    /** \cond INTERNAL */
    typedef GaussianFilter<ProcessModel, ObsrvModel, PointSetTransform> This;
    typedef typename Traits<This>::KalmanGain KalmanGain;
    typedef typename Traits<This>::StateNoise StateNoise;
    typedef typename Traits<This>::ObsrvNoise ObsrvNoise;
    typedef typename Traits<This>::StatePointSet StatePointSet;
    typedef typename Traits<This>::ObsrvPointSet ObsrvPointSet;
    typedef typename Traits<This>::StateNoisePointSet StateNoisePointSet;
    typedef typename Traits<This>::ObsrvNoisePointSet ObsrvNoisePointSet;
    /** \endcond */

public:
    /* public concept interface types */
    typedef typename Traits<This>::State State;    
    typedef typename Traits<This>::Input Input;
    typedef typename Traits<This>::Observation Obsrv;    
    typedef typename Traits<This>::StateDistribution StateDistribution;    

public:
    /**
     * Creates a Gaussian filter
     *
     * @param process_model         Process model instance
     * @param Obsrv_model           Obsrv model instance
     * @param point_set_transform   Point set tranfrom such as the unscented
     *                              transform
     */
    GaussianFilter(const std::shared_ptr<ProcessModel>& process_model,
                   const std::shared_ptr<ObsrvModel>& Obsrv_model,
                   const std::shared_ptr<PointSetTransform>& point_set_transform)
        : process_model_(process_model),
          obsrv_model_(Obsrv_model),
          point_set_transform_(point_set_transform),
          /*
           * Set the augmented Gaussian dimension.
           *
           * The global dimension is dimension of the augmented Gaussian which
           * consists of state Gaussian, state noise Gaussian and the
           * observation noise Gaussian.
           */
          global_dimension_(process_model_->state_dimension()
                            + process_model_->noise_dimension()
                            + obsrv_model_->noise_dimension()),
          /*
           * Initialize the points-set Gaussian (e.g. sigma points) of the
           * \em state noise. The number of points is determined by the
           * augmented Gaussian with the dimension  global_dimension_
           */
          X_Q(process_model_->noise_dimension(),
              PointSetTransform::number_of_points(global_dimension_)),

          /*
           * Initialize the points-set Gaussian (e.g. sigma points) of the
           * \em observation noise. The number of points is determined by the
           * augmented Gaussian with the dimension global_dimension_
           */
          X_R(obsrv_model_->noise_dimension(),
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
        point_set_transform_->forward(
            fl::Gaussian<StateNoise>(process_model_->noise_dimension()),
            global_dimension_,
            process_model_->state_dimension(),
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
        point_set_transform_->forward(
            fl::Gaussian<ObsrvNoise>(obsrv_model_->noise_dimension()),
            global_dimension_,
            process_model_->state_dimension()
            + process_model_->noise_dimension(),
            X_R);

        /*
         * Setup the point set of the observation predictions
         */
        const size_t point_count =
                PointSetTransform::number_of_points(global_dimension_);
        X_y.resize(point_count);
        X_y.dimension(obsrv_model_->obsrv_dimension());
    }

    /**
     * \copydoc FilterInterface::predict
     */
    virtual void predict(double delta_time,
                         const Input& input,
                         const StateDistribution& prior_dist,
                         StateDistribution& predicted_dist)
    {
        std::cout << "-> predicting ... " << std::endl;

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
        point_set_transform_->forward(prior_dist,
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
            X_r.point(i, process_model_->predict_state(delta_time,
                                                       X_r.point(i),
                                                       X_Q.point(i),
                                                       input));
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
        auto&& X = X_r.centered_points();

        /*
         * Obtain the weights of point as a vector
         *
         * W = [w_cov[1]  w_cov[2]  ... w_cov[n]]
         *
         * Note that the covariance weights are used.
         */
        auto&& W = X_r.covariance_weights_vector();

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
        predicted_dist.Mean(X_r.mean());
        predicted_dist.Covariance(X * W.asDiagonal() * X.transpose());
    }

    /**
     * \copydoc FilterInterface::update
     */
    virtual void update(const Obsrv& y,
                        const StateDistribution& predicted_dist,
                        StateDistribution& posterior_dist)
    {
        point_set_transform_->forward(predicted_dist,
                                      global_dimension_,
                                      0,
                                      X_r);

        const size_t point_count = X_r.count_points();
        for (size_t i = 0; i < point_count; ++i)
        {
            X_y.point(i, obsrv_model_->predict_observation(X_r.point(i),
                                                           X_R.point(i)));
        }

        auto W = X_r.covariance_weights_vector();
        auto X = X_r.centered_points();
        auto Y = X_y.centered_points();

        auto cov_xx = X * W.asDiagonal() * X.transpose();
        auto cov_yy = Y * W.asDiagonal() * Y.transpose();
        auto cov_xy = X * W.asDiagonal() * Y.transpose();

        const KalmanGain& K = cov_xy * cov_yy.inverse();

        posterior_dist.Mean(X_r.mean() + K * (y - X_y.mean()));
        posterior_dist.Covariance(cov_xx - K * cov_yy * K.transpose());
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

protected:
    std::shared_ptr<ProcessModel> process_model_;
    std::shared_ptr<ObsrvModel> obsrv_model_;
    std::shared_ptr<PointSetTransform> point_set_transform_;

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
};

}

#endif
