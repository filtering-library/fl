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
 * \file sigma_point_additive_prediction_policy.hpp
 * \date July 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once


#include <Eigen/Dense>

#include <fl/util/meta.hpp>
#include <fl/util/traits.hpp>
#include <fl/util/descriptor.hpp>
#include <fl/filter/gaussian/transform/point_set.hpp>
#include <fl/filter/gaussian/quadrature/sigma_point_quadrature.hpp>

namespace fl
{

// Forward declarations
template <typename...> class SigmaPointPredictPolicy;

template <
    typename SigmaPointQuadrature,
    typename AdditiveTransitionFunction
>
class SigmaPointPredictPolicy<
          SigmaPointQuadrature,
          Additive<AdditiveTransitionFunction>>
    : public Descriptor
{
public:
    typedef typename AdditiveTransitionFunction::State State;
    typedef typename AdditiveTransitionFunction::Input Input;

    enum : signed int
    {
        NumberOfPoints = SigmaPointQuadrature
                            ::number_of_points(SizeOf<State>::Value)
    };

    typedef PointSet<State, NumberOfPoints> StatePointSet;

    template <
        typename Belief
    >
    void operator()(const AdditiveTransitionFunction&
                              additive_state_transition_function,
                    const SigmaPointQuadrature& quadrature,
                    const Belief& prior_belief,
                    const Input& u,
                    Belief& predicted_belief)
    {
        auto f = [&](const State& x)
        {
            return additive_state_transition_function.expected_state(x, u);
        };

        quadrature.propergate_gaussian(f, prior_belief, Y, Z);

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
        auto X_c = Z.centered_points();

        /*
         * Obtain the weights of point as a vector
         *
         * W = [w_cov[1]  w_cov[2]  ... w_cov[n]]
         *
         * Note that the covariance weights are used.
         */
        auto W = Z.covariance_weights_vector();

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
        predicted_belief.dimension(prior_belief.dimension());
        predicted_belief.mean(Z.mean());
        predicted_belief.covariance(
            X_c * W.asDiagonal() * X_c.transpose()
            + additive_state_transition_function.noise_covariance());
    }

    virtual std::string name() const
    {
        return "SigmaPointPredictPolicy<"
                + this->list_arguments(
                       "SigmaPointQuadrature",
                       "Additive<AdditiveTransitionFunction>")
                + ">";
    }

    virtual std::string description() const
    {
        return "Sigma Point based filter prediction policy for state "
               "transition model with additive noise";
    }

protected:
    StatePointSet Y;
    StatePointSet Z;
};

}


