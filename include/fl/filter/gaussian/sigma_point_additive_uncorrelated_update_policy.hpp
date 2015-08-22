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
 * \file sigma_point_additive_uncorrelated_update_policy.hpp
 * \date July 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__FILTER__GAUSSIAN__SIGMA_POINT_ADDITIVE_UNCORRELATED_UPDATE_POLICY_HPP
#define FL__FILTER__GAUSSIAN__SIGMA_POINT_ADDITIVE_UNCORRELATED_UPDATE_POLICY_HPP

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
    typename AdditiveUncorrelatedObsrvFunction
>
class SigmaPointUpdatePolicy<
          SigmaPointQuadrature,
          AdditiveUncorrelated<AdditiveUncorrelatedObsrvFunction>>
    : public Descriptor
{
public:
    typedef typename AdditiveUncorrelatedObsrvFunction::State State;
    typedef typename AdditiveUncorrelatedObsrvFunction::Obsrv Obsrv;

    enum : signed int
    {
        NumberOfPoints =
            SigmaPointQuadrature::number_of_points(SizeOf<State>::Value)
    };

    // static_assert(false, "Just implementing this ........");

    typedef PointSet<State, NumberOfPoints> StatePointSet;
    typedef PointSet<Obsrv, NumberOfPoints> ObsrvPointSet;

    template <
        typename Belief
    >
    void operator()(const AdditiveUncorrelatedObsrvFunction& obsrv_function,
                    const SigmaPointQuadrature& quadrature,
                    const Belief& prior_belief,
                    const Obsrv& obsrv,
                    Belief& posterior_belief)
    {
        auto&& h = [=](const State& x)
        {
           return obsrv_function.expected_observation(x);
        };

        quadrature.propergate_gaussian(h, prior_belief, X, Z);

        auto R_inv =
            obsrv_function
                .noise_diagonal_covariance()
                .diagonal()
                .cwiseInverse()
                .eval();

        auto W_inv =
            X.covariance_weights_vector()
                .cwiseInverse()
                .eval();

        auto&& prediction = Z.center();
        auto&& Y_c = Z.points();
        auto&& X_c = X.centered_points();

        auto innovation = (obsrv - prediction).eval();

        auto C = (Y_c.transpose() * R_inv.asDiagonal() * Y_c).eval();
        C += W_inv.asDiagonal();
        C = C.inverse();

        auto correction = (
           X_c * C * Y_c.transpose() *  R_inv.asDiagonal() * innovation).eval();

        posterior_belief.mean(X.mean() + correction);
        posterior_belief.covariance(X_c * C * X_c.transpose());
    }

    virtual std::string name() const
    {
        return "SigmaPointUpdatePolicy<"
            + this->list_arguments(
                 "SigmaPointQuadrature",
                 "AdditiveUncorrelated<AdditiveUncorrelatedObservationFunction>")
            + ">";
    }

    virtual std::string description() const
    {
        return "Sigma Point based filter update policy for observation model"
               " with additive uncorrelated noise";
    }

protected:
    StatePointSet X;
    ObsrvPointSet Z;
};

}

#endif

