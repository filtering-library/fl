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
 * \file gaussian_filter_ukf_npn_aon.hpp
 * \date January 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__FILTER__GAUSSIAN__GAUSSIAN_FILTER_UKF_NNPN_AON_HPP
#define FL__FILTER__GAUSSIAN__GAUSSIAN_FILTER_UKF_NNPN_AON_HPP

#include <map>
#include <tuple>
#include <memory>

#include <fl/util/meta.hpp>
#include <fl/util/traits.hpp>

#include <fl/exception/exception.hpp>
#include <fl/filter/filter_interface.hpp>
#include <fl/filter/gaussian/point_set.hpp>

namespace fl
{

template <typename...> class GaussianFilter;

enum AdditiveObservationNoise { };

/**
 * GaussianFilter Traits
 */
template <typename ProcessModel,
          typename ObservationModel,
          typename PointSetTransform>
struct Traits<
           GaussianFilter<
               ProcessModel,
               ObservationModel,
               PointSetTransform,
               AdditiveObservationNoise>>
{
    typedef GaussianFilter<
                ProcessModel,
                ObservationModel,
                PointSetTransform,
                AdditiveObservationNoise
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
    typedef std::shared_ptr<Filter> Ptr;
    typedef typename Traits<ProcessModel>::State State;
    typedef typename Traits<ProcessModel>::Input Input;    
    typedef typename Traits<ObservationModel>::Observation Observation;

    /**
     * Represents the underlying distribution of the estimated state. In
     * case of a Point Based Kalman filter, the distribution is a simple
     * Gaussian with the dimension of the \c State.
     */
    typedef Gaussian<State> StateDistribution;

    /** \cond INTERNAL */
    typedef typename Traits<ProcessModel>::Noise StateNoise;
    typedef typename Traits<ObservationModel>::Noise ObsrvNoise;

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
    static constexpr int NumberOfPoints = PointSetTransform::number_of_points(
                        JoinSizes<
                            State::RowsAtCompileTime,
                            StateNoise::RowsAtCompileTime
                        >::Size);

    typedef PointSet<State, NumberOfPoints> StatePointSet;
    typedef PointSet<Observation, NumberOfPoints> ObsrvPointSet;
    typedef PointSet<StateNoise, NumberOfPoints> StateNoisePointSet;

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
 * Gaussian for Non-Additive Process Noise & Additive Observation Noise
 *
 * \tparam ProcessModel
 * \tparam ObservationModel
 *
 * \ingroup filters
 * \ingroup sigma_point_kalman_filters
 * \ingroup unscented_kalman_filter
 */
template<
    typename ProcessModel,
    typename ObservationModel,
    typename PointSetTransform>
class GaussianFilter<
          ProcessModel,
          ObservationModel,
          PointSetTransform,
          AdditiveObservationNoise>
    :
    /* Implement the conceptual filter interface */
    public FilterInterface<
              GaussianFilter<
                  ProcessModel,
                  ObservationModel,
                  PointSetTransform,
                  AdditiveObservationNoise>>

{
protected:
    /** \cond INTERNAL */
    typedef GaussianFilter<
                ProcessModel,
                ObservationModel,
                PointSetTransform,
                AdditiveObservationNoise
            > This;

    typedef typename Traits<This>::KalmanGain KalmanGain;
    typedef typename Traits<This>::StateNoise StateNoise;
    typedef typename Traits<This>::ObsrvNoise ObsrvNoise;
    typedef typename Traits<This>::StatePointSet StatePointSet;
    typedef typename Traits<This>::ObsrvPointSet ObsrvPointSet;
    typedef typename Traits<This>::StateNoisePointSet StateNoisePointSet;
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
     * \param process_model         Process model instance
     * \param obsrv_model           Obsrv model instance
     * \param point_set_transform   Point set tranfrom such as the unscented
     *                              transform
     */
    GaussianFilter(const std::shared_ptr<ProcessModel>& process_model,
                   const std::shared_ptr<ObservationModel>& obsrv_model,
                   const std::shared_ptr<PointSetTransform>& point_set_transform)
        : process_model_(process_model),
          obsrv_model_(obsrv_model),
          point_set_transform_(point_set_transform),
          /*
           * Set the augmented Gaussian dimension.
           *
           * The global dimension is dimension of the augmented Gaussian which
           * consists of state Gaussian, state noise Gaussian and the
           * observation noise Gaussian.
           */
          global_dimension_(process_model_->state_dimension()
                            + process_model_->noise_dimension()),
          /*
           * Initialize the points-set Gaussian (e.g. sigma points) of the
           * \em state noise. The number of points is determined by the
           * augmented Gaussian with the dimension  global_dimension_
           */
          X_Q(process_model_->noise_dimension(),
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
            Gaussian<StateNoise>(process_model_->noise_dimension()),
            global_dimension_,
            process_model_->state_dimension(),
            X_Q);

        /*
         * Setup the point set of the observation predictions
         */
        const size_t point_count =
                PointSetTransform::number_of_points(global_dimension_);

        X_y.resize(point_count);
        X_y.dimension(obsrv_model_->observation_dimension());

        /*
         * Setup the point set of the state predictions
         */
        X_r.resize(point_count);
        X_r.dimension(process_model_->state_dimension());
    }

    /**
     * \copydoc FilterInterface::predict
     */
    virtual void predict(double delta_time,
                         const Input& input,
                         const StateDistribution& prior_dist,
                         StateDistribution& predicted_dist)
    {
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
        X = X_r.centered_points();

        /*
         * Obtain the weights of point as a vector
         *
         * W = [w_cov[1]  w_cov[2]  ... w_cov[n]]
         *
         * Note that the covariance weights are used.
         */
        W = X_r.covariance_weights_vector();

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
        predicted_dist.mean(X_r.mean());
        predicted_dist.covariance(X * W.asDiagonal() * X.transpose());
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
                                                           0 /* delta time */));
        }

        fl::invert_diagonal_Vector(X_r.covariance_weights_vector(), inv_W);
        fl::invert_diagonal_Vector(obsrv_model_->noise_covariance_vector(), inv_R);

        X = X_r.centered_points();
        Y = X_y.centered_points();

        prediction = X_y.mean();
        innovation = (y - prediction);

        for (int i = 0; i < y.rows(); ++i)
        {
            if (std::abs(innovation(i, 0)) > threshold)
            {
                inv_R(i, 0) = 1./inv_sigma;
            }
        }

        C  = (Y.transpose() * inv_R.asDiagonal() * Y).eval();
        C += inv_W.asDiagonal();
        C  = C.inverse();

        correction = X * C * Y.transpose() * inv_R.asDiagonal() * innovation;

        posterior_dist.mean(predicted_dist.mean() +  correction);
        posterior_dist.covariance(X * C * X.transpose());
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

    const std::shared_ptr<ProcessModel>& process_model()
    {
        return process_model_;
    }

    const std::shared_ptr<ObservationModel>& observation_model()
    {
        return obsrv_model_;
    }

    const std::shared_ptr<ProcessModel>& point_set_transform()
    {
        return point_set_transform_;
    }

public:
    double threshold;
    double inv_sigma;

protected:
    std::shared_ptr<ProcessModel> process_model_;
    std::shared_ptr<ObservationModel> obsrv_model_;
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
    /** \endcond */

public:
    /** \cond INTERNAL */
    /* Dungeon - keep put! */
    decltype(X_y.mean()) prediction;
    decltype(prediction) innovation;
    decltype(X_r.mean()) correction;

    decltype(X_r.centered_points()) X;
    decltype(X_y.centered_points()) Y;
    decltype(X_r.covariance_weights_vector()) W;
    decltype(X_r.covariance_weights_vector()) inv_W;
    decltype(obsrv_model_->noise_covariance_vector()) inv_R;
    typename std::remove_const<
        decltype((Y.transpose() * inv_R.asDiagonal() * Y).eval())
    >::type C;
    /** \endcond */
};

}

#endif
