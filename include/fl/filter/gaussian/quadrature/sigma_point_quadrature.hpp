/*
 * This is part of the FL library, a C++ Bayesian filtering library
 * (https://github.com/filtering-library)
 *
 * Copyright (c) 2014 Jan Issac (jan.issac@gmail.com)
 * Copyright (c) 2014 Manuel Wuthrich (manuel.wuthrich@gmail.com)
 *
 * This Source Code Form is subject to the terms of the MIT License (MIT).
 * A copy of the license can be found in the LICENSE file distributed with this
 * source code.
 */

/**
 * \file sigma_point_quadrature.hpp
 * \date July 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__FILTER__GAUSSIAN__SIGMA_POINT_QUADRATURE_HPP
#define FL__FILTER__GAUSSIAN__SIGMA_POINT_QUADRATURE_HPP

#include <fl/util/meta.hpp>
#include <fl/util/traits.hpp>
#include <fl/util/descriptor.hpp>
#include <fl/distribution/gaussian.hpp>
#include <fl/filter/gaussian/transform/point_set.hpp>

namespace fl
{

// Forward declaration
template <typename ... > class SigmaPointQuadrature;

/**
 * \ingroup numeric_integration
 *
 * \brief Represents a numeric integration tool based on sigma point selections
 *        of Gaussian distribution
 *
 * ## Examples: Expectation computation ##
 *
 * ### Computing the first moment (mean) ####
 * \f$\mathbb{E}[f(x)] = \displaystyle\int f(x) p(x) \mathrm{d}x\f$
 *
 * \code
 * int some_function(int x) { return ...; }
 *
 * ...
 *
 * // setup a some gaussian
 * typedef Eigen::Vector<Real, 10> MyVariate;
 * Gaussian<MyVariate> p;
 *
 * p.mean(MyVariate::Random(10));
 * p.covariance(p.covariance() * std::rand());
 *
 * // create the lambda we will pass to the integrator
 * auto f = [=](int x) { return some_function(x); };
 *
 * auto f_mean = integrate(f, p);
 * \endcode
 *
 * ### Computing the second centered moment (covariance) ###
 * \f$\begin{align} \mathbb{Cov}[f(x)]
 *      & = \mathbb{E}[(f(x)-\mathbb{E}[f(x)])^2] \\
 *      & = \displaystyle\int
 *          \underbrace{\left(f(x)-\mu_f\right)\left(f(x)-\mu_f\right)^T}_{g(x):=}
 *          p(x) \mathrm{d}x \\
 *      & = \displaystyle\int g(x) p(x) \mathrm{d}x
 * \end{align}
 * \f$
 *
 * \code
 * // ... continuing the code from the previous example above
 *
 * // create the quadratic form (f(x)-mu_f)(f(x)-mu_f)^T lambda we will pass
 * // to the integrator
 * auto g = [=](int x) { return ((some_function(x) - f_mean) *
 *                               (some_function(x) - f_mean).transpose()).eval(); };
 * auto f_covariance = integrate(g, p);
 *
 *
 * \endcode
 */
template <typename Transform>
class SigmaPointQuadrature<Transform>
    : Descriptor
{
public:
    /**
     * \brief Creates a SigmaPointQuadrature
     *
     * \param transform     Transform instance, e.g. UnscentedTrasnform or
     *                      MonteCarloTransform
     */
    explicit SigmaPointQuadrature(const Transform& transform)
        : transform_(transform)
    { }

    /**
     * \brief Integration function performing integrals of the form
     *        \f$
     *         \displaystyle\int f(x) p(x) \mathrm{d}x
     *        \f$.
     *
     * \f$p(x)\f$ is a Gaussian distribution \f${\cal N}(x; \mu_x, \Sigma_x)\f$.
     *
     * \tparam Integrand    \a Integrand function type of \f$f(x)\f$.
     *                      \a Integrand is either a \c lambda or a functor
     * \tparam Gaussian     Gaussian distribuion type of \f$p(x)\f$
     *
     * \return integration result which has the type of the specified Integrand.
     *
     * \note   This integrator is quite generic. However, it can be quite
     *         inefficient since it doesn't make any assumptions on the
     *         integration result type. It is recommanded to be used in
     *         educational and prototyping purposes only.
     *         Use integrate_to_points() or integrate_to_gaussian() for higher
     *         efficiency.
     */
    template <
        typename Integrand,
        typename Gaussian,
    >
    auto integrate(Integrand f, const Gaussian& distr)
    -> decltype(f(distr.mean()))
    {
        const int point_count = Transform::number_of_points(distr.dimension());

        typedef typename Gaussian::Variate Variate;
        auto X = PointSet<
                     Variate,
                     Transform::number_of_points(SizeOf<Variate>::Value)
                >(distr.dimension(), point_count);

        transform_(distr, X);

        auto E = f(X[0]);
        E *= X.weight(0);

        for (int i = 1; i < point_count; ++i)  E += X.weight(i) * f(X[i]);

        return E;
    }

    /**
     * \brief Integration function performing integrals of the form
     *        \f$
     *         \displaystyle\int f(x, y) p(y\mid x) p(x) \mathrm{d}y\mathrm{d}x
     *        \f$.
     *
     * \f$p(x)\f$ and \f$p(y\mid x)\f$ are Gaussian distributions
     * \f${\cal N}(x; \mu_x, \Sigma_x)\f$ and
     * \f${\cal N}(y; \mu_{y|x}, \Sigma_{y|x})\f$, repsectively
     *
     * \tparam Integrand        Integrand function type of \f$f(x, y)\f$
     *
     * \return integration result which has the type of the specified Integrand.
     *
     * \note   This integrator is quite generic. However, it can be quite
     *         inefficient since it doesn't make any assumptions on the
     *         integration result type. It is recommanded to be used in
     *         educational and prototyping purposes only.
     *         Use integrate_to_points() or integrate_to_gaussian() for higher
     *         efficiency.
     */
    template <
        typename Integrand,
        typename ModelGaussian,
        typename PriorGaussian
    >
    auto integrate(
            Integrand f,
            const ModelGaussian& model_distr,
            const PriorGaussian& prior_distr)
    -> decltype(f(model_distr.mean(), prior_distr.mean()))
    {
        typedef typename ModelGaussian::Variate VariateA;
        typedef typename PriorGaussian::Variate VariateB;

        enum : signed int
        {
            NumberOfPoints = Transform::number_of_points(
                                 JoinSizes<
                                     SizeOf<VariateA>::Value,
                                     SizeOf<VariateB>::Value
                                 >::Size)
        };

        typedef PointSet<VariateA, NumberOfPoints> PointSetA;
        typedef PointSet<VariateB, NumberOfPoints> PointSetB;

        const int augmented_dim = model_distr.dimension() + prior_distr.dimension();
        const int point_count = Transform::number_of_points(augmented_dim);

        PointSetA X_a(model_distr.dimension(), point_count);
        PointSetB X_b(prior_distr.dimension(), point_count);

        transform_(model_distr, augmented_dim, 0, X_a);
        transform_(prior_distr, augmented_dim, model_distr.dimension(), X_b);

        auto E = f(X_a[0], X_b[0]);
        E *= X_a.weight(0);

        for (int i = 1; i < point_count; ++i)
        {
            E += X_a.weight(i) * f(X_a[i], X_b[i]);
        }

        return E;
    }

    template <
        typename Integrand,
        typename ModelGaussian,
        typename PriorGaussian,
        typename ModelSigmaPointsSet,
        typename PriorSigmaPointsSet,
        typename PosteriorSigmaPointsSet
    >
    void integrate_to_points(
            Integrand f,
            const ModelGaussian& model_distr,
            const PriorGaussian& prior_distr,
            ModelSigmaPointsSet& X,
            PriorSigmaPointsSet& Y,
            PosteriorSigmaPointsSet& Z) const
    {
        int augmented_dim = model_distr.dimension() + prior_distr.dimension();
        int point_count = Transform::number_of_points(augmented_dim);

        X.resize(model_distr.dimension(), point_count);
        Y.resize(prior_distr.dimension(), point_count);

        transform_(model_distr, augmented_dim, 0, X);
        transform_(prior_distr, augmented_dim, model_distr.dimension(), Y);

        auto p0 = f(X[0], Y[0]);
        Z.resize(p0.size(), point_count);
        Z.point(0, p0, X.weights(0).w_mean, X.weights(0).w_cov);

        for (int i = 1; i < point_count; ++i)
        {
            Z.point(i, f(X[i], Y[i]), X.weights(i).w_mean, X.weights(i).w_cov);
        }
    }

    template <
        typename Integrand,
        typename ModelGaussian,
        typename PriorGaussian,
        typename PosteriorSigmaPointsSet
    >
    void integrate_to_points(
            Integrand f,
            const ModelGaussian& model_distr,
            const PriorGaussian& prior_distr,
            PosteriorSigmaPointsSet& Z) const
    {
        typedef typename ModelGaussian::Variate ModelVariate;
        typedef typename PriorGaussian::Variate PriorVariate;

        enum : signed int
        {
            NumberOfPoints = number_of_points(
                                 JoinSizes<
                                     SizeOf<ModelVariate>::Value,
                                     SizeOf<PriorVariate>::Value
                                 >::Size)
        };

        typedef PointSet<ModelVariate, NumberOfPoints> ModelSigmaPointsSet;
        typedef PointSet<PriorVariate, NumberOfPoints> PriorSigmaPointsSet;

        int augmented_dim = model_distr.dimension() + prior_distr.dimension();
        int point_count = Transform::number_of_points(augmented_dim);

        auto X = ModelSigmaPointsSet(model_distr.dimension(), point_count);
        auto Y = PriorSigmaPointsSet(prior_distr.dimension(), point_count);

        integrate_to_points(f, model_distr, prior_distr, X, Y, Z);
    }

    template <
        typename Integrand,
        typename PriorGaussian,
        typename PriorSigmaPointsSet,
        typename PosteriorSigmaPointsSet
    >
    void integrate_to_points(
            Integrand f,
            const PriorGaussian& prior_distr,
            PriorSigmaPointsSet& Y,
            PosteriorSigmaPointsSet& Z) const
    {
        int point_count = Transform::number_of_points(prior_distr.dimension());

        Y.resize(prior_distr.dimension(), point_count);

        transform_(prior_distr, Y);

        auto p0 = f(Y[0]);
        Z.resize(p0.size(), point_count);
        Z.point(0, p0, Y.weights(0).w_mean, Y.weights(0).w_cov);

        for (int i = 1; i < point_count; ++i)
        {
            Z.point(i, f(Y[i]), Y.weights(i).w_mean, Y.weights(i).w_cov);
        }
    }

    template <
        typename Integrand,
        typename ModelGaussian,
        typename PriorGaussian,
        typename APostGaussian
    >
    void integrate_to_gaussian(
            Integrand f,
            const ModelGaussian& model_distr,
            const PriorGaussian& prior_distr,
            APostGaussian& apost_distr) const
    {
        typedef typename ModelGaussian::Variate ModelVariate;
        typedef typename PriorGaussian::Variate PriorVariate;
        typedef typename APostGaussian::Variate APostVariate;

        enum : signed int
        {
            NumberOfPoints = number_of_points(
                                 JoinSizes<
                                     SizeOf<ModelVariate>::Value,
                                     SizeOf<PriorVariate>::Value
                                 >::Size)
        };

        typedef PointSet<ModelVariate, NumberOfPoints> ModelSigmaPointsSet;
        typedef PointSet<PriorVariate, NumberOfPoints> PriorSigmaPointsSet;
        typedef PointSet<APostVariate, NumberOfPoints> APostSigmaPointsSet;

        int augmented_dim = model_distr.dimension() + prior_distr.dimension();
        int point_count = Transform::number_of_points(augmented_dim);

        auto X = ModelSigmaPointsSet(model_distr.dimension(), point_count);
        auto Y = PriorSigmaPointsSet(prior_distr.dimension(), point_count);
        auto Z = APostSigmaPointsSet(apost_distr.dimension(), point_count);

        integrate_to_points(f, model_distr, prior_distr, X, Y, Z);

        apost_distr.mean(Z.center());
        auto&& Z_c = Z.points();
        auto&& W = X.covariance_weights_vector();
        apost_distr.covariance(Z_c * W.asDiagonal() * Z_c.transpose());
    }

    /**
     * \brief Returns a reference to the trandform used internally to generate
     *        the points or samples from the gaussian distributions
     */
    Transform& transform()
    {
        return transform_;
    }

    /**
     * \brief Returns a const reference to the trandform used internally to
     *        generate the points or samples from the gaussian distributions
     */
    const Transform& transform() const
    {
        return transform_;
    }

    /**
     * \return Number of points generated by transform
     *
     * \param dimension Dimension of the Gaussian
     */
    static constexpr int number_of_points(int dimension)
    {
        return Transform::number_of_points(dimension);
    }

    virtual std::string name() const
    {
        return "SigmaPointQuadrature<"
                    + list_arguments(transform().name())
                + ">";
    }

    virtual std::string description() const
    {
        return "Sigma Point based numerical quadrature using :"
                + indent(transform().description());
    }

protected:
    Transform transform_;
};

}

#endif
