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
 * \file sigma_point_update_policy.hpp
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
template <typename...> class SigmaPointUpdatePolicy;


template <
    typename SigmaPointQuadrature,
    typename ObservationFunction
>
class SigmaPointUpdatePolicy<
          SigmaPointQuadrature,
          ObservationFunction>
    : public SigmaPointUpdatePolicy<
                SigmaPointQuadrature,
                NonAdditive<ObservationFunction>>
{ };


template <
    typename SigmaPointQuadrature,
    typename ObservationFunction
>
class SigmaPointUpdatePolicy<
          SigmaPointQuadrature,
          NonAdditive<ObservationFunction>>
    : public Descriptor
{
public:
    typedef typename ObservationFunction::State State;
    typedef typename ObservationFunction::Obsrv Obsrv;
    typedef typename ObservationFunction::Noise Noise;

    enum : signed int
    {
        NumberOfPoints = SigmaPointQuadrature::number_of_points(
                             JoinSizes<
                                 SizeOf<State>::Value,
                                 SizeOf<Noise>::Value
                             >::Size)
    };

    typedef PointSet<State, NumberOfPoints> StatePointSet;
    typedef PointSet<Noise, NumberOfPoints> NoisePointSet;
    typedef PointSet<Obsrv, NumberOfPoints> ObsrvPointSet;

    template <
        typename Belief
    >
    void operator()(const ObservationFunction& obsrv_function,
                    const SigmaPointQuadrature& quadrature,
                    const Belief& prior_belief,
                    const Obsrv& obsrv,
                    Belief& posterior_belief)
    {
        // static_assert() is non-additive

        noise_distr_.dimension(obsrv_function.noise_dimension());

        auto&& h = [&](const State& x, const Noise& w)
        {
           return obsrv_function.observation(x, w);
        };

        quadrature.propergate_gaussian(h, prior_belief, noise_distr_, X, Y, Z);

        auto&& prediction = Z.center();
        auto&& Z_c = Z.points();
        auto&& W = X.covariance_weights_vector();
        auto&& X_c = X.centered_points();

        auto innovation = (obsrv - prediction).eval();
        auto cov_xx = (X_c * W.asDiagonal() * X_c.transpose()).eval();
        auto cov_yy = (Z_c * W.asDiagonal() * Z_c.transpose()).eval();
        auto cov_xy = (X_c * W.asDiagonal() * Z_c.transpose()).eval();
        auto cov_yx = cov_xy.transpose().eval();

        auto x_updated = (X.mean() + cov_xy * solve(cov_yy, innovation)).eval();
        auto cov_xx_updated = (cov_xx - cov_xy * solve(cov_yy, cov_yx)).eval();

        posterior_belief.dimension(prior_belief.dimension());
        posterior_belief.mean(x_updated);
        posterior_belief.covariance(cov_xx_updated);

        //        auto K = (cov_xy * cov_yy.inverse()).eval();
        //        posterior_belief.mean(X.mean() + K * innovation);
        //        posterior_belief.covariance(cov_xx - K * cov_yy * K.transpose());
    }

    virtual std::string name() const
    {
        return "SigmaPointUpdatePolicy<"
                + this->list_arguments(
                       "SigmaPointQuadrature",
                       "NonAdditive<ObservationFunction>")
                + ">";
    }

    virtual std::string description() const
    {
        return "Sigma Point based filter update policy for observation model"
               " with non-additive noise";
    }


protected:
    StatePointSet X;
    NoisePointSet Y;
    ObsrvPointSet Z;
    Gaussian<Noise> noise_distr_;
};

}


