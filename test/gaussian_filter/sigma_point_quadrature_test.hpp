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
 * \file sigma_point_quadrature_test.hpp
 * \date August 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>
#include "../typecast.hpp"

#include <Eigen/Dense>

#include <random>

#include <fl/util/profiling.hpp>
#include <fl/util/math.hpp>
#include <fl/distribution/gaussian.hpp>
#include <fl/filter/gaussian/transform/point_set.hpp>
#include <fl/filter/gaussian/transform/unscented_transform.hpp>
#include <fl/filter/gaussian/transform/monte_carlo_transform.hpp>
#include <fl/filter/gaussian/quadrature/sigma_point_quadrature.hpp>

template <typename TestType>
class SigmaPointQuadratureTests
    : public testing::Test
{
public:
    typedef typename TestType::Parameter Configuration;

    enum: signed int
    {
        DimA = Configuration::DimA,
        DimB = Configuration::DimB,

        SizeA = fl::TestSize<DimA, TestType>::Value,
        SizeB = fl::TestSize<DimB, TestType>::Value
    };

    typedef Eigen::Matrix<fl::Real, SizeA, 1> VariateA;
    typedef Eigen::Matrix<fl::Real, SizeB, 1> VariateB;

    typedef fl::Gaussian<VariateA> GaussianA;
    typedef fl::Gaussian<VariateB> GaussianB;

    typedef typename GaussianA::SecondMoment SecondMomentA;
    typedef typename GaussianB::SecondMoment SecondMomentB;

    typedef Eigen::Matrix<fl::Real, SizeA, SizeA> MatrixAA;
    typedef Eigen::Matrix<fl::Real, SizeB, SizeA> MatrixAB;

    typedef typename Configuration::TransformSelection::Transform Transform;
    typedef fl::SigmaPointQuadrature<Transform> Quadrature;

    SigmaPointQuadratureTests()
        : F(MatrixAA::Random(DimA, DimA)),
          H(MatrixAB::Random(DimB, DimA)),
          p_A(DimA),
          p_B(DimB),
          quadrature(Transform()),
          eps(Configuration::TransformSelection::epsilon)
    {
        // create a random source Gaussians
        p_A.mean(VariateA::Random(DimA));
        p_A.covariance(p_A.covariance() * fl::Real(std::rand()) / RAND_MAX);

        p_B.mean(VariateB::Random(DimB));
        p_B.covariance(p_B.covariance() * fl::Real(std::rand()) / RAND_MAX);
    }

    template <typename Var> Var f(const Var& x)
    {
        return F * x;
    }

    template <typename VarA, typename VarB>
    VarB f(const VarA& x, const VarB& w)
    {
        return H * x + w;
    }

    void integrate_fx_px()
    {
        using namespace fl;

        // compute the expected integration result analytically
        auto expect_gaussian = GaussianA(DimA);
        expect_gaussian.mean(f(p_A.mean()));
        expect_gaussian.covariance(F * p_A.covariance() * F.transpose());

        // create the gaussian which will contail the integration results
        auto result_gaussian = GaussianA(DimA);

        // define the expected mean lambda function
        // here we use a simple identity function
        auto mean_f = [&] (const VariateA& x) -> VariateA { return f(x); };

        // define the expected covariance lambda function
        auto cov_f = [&] (const VariateA& x) -> SecondMomentA
        {
            return (f(x) - result_gaussian.mean()) *
                   (f(x) - result_gaussian.mean()).transpose();
        };

        // EXPECT_FALSE(result_gaussian.is_approx(expect_gaussian, eps, true));

        // integrate mean and covariance
        result_gaussian.mean(quadrature.integrate(mean_f, p_A));
        result_gaussian.covariance(quadrature.integrate(cov_f, p_A));

        EXPECT_TRUE(result_gaussian.is_approx(expect_gaussian, eps, true));
    }

    void integrate_fxy_pxy()
    {
        using namespace fl;

        auto expect_gaussian = GaussianB(DimB);
        auto result_gaussian = GaussianB(DimB);

        expect_gaussian.mean(f(p_A.mean(), p_B.mean()));
        expect_gaussian.covariance(
            H * p_A.covariance() * H.transpose() + p_B.covariance());


        // define the expected mean lambda function
        // here we use a simple identity function
        auto mean_f = [&] (const VariateA& x, const VariateB& y) -> VariateB
        {
            return f(x, y);
        };

        // define the expected covariance lambda function
        auto cov_f = [&] (const VariateA& x, const VariateB& y) -> SecondMomentB
        {
            return (f(x, y) - result_gaussian.mean()) *
                   (f(x, y) - result_gaussian.mean()).transpose();
        };

        // EXPECT_FALSE(result_gaussian.is_approx(expect_gaussian, eps, true));

        // integrate mean and covariance
        result_gaussian.mean(quadrature.integrate(mean_f, p_A, p_B));
        result_gaussian.covariance(quadrature.integrate(cov_f, p_A, p_B));

        EXPECT_TRUE(result_gaussian.is_approx(expect_gaussian, eps, true));
    }

    void propergate_gaussian_Z()
    {
        using namespace fl;

        auto result_gaussian = GaussianA(DimA);
        auto expect_gaussian = GaussianA(DimA);
        expect_gaussian.mean(f(p_A.mean()));
        expect_gaussian.covariance(F * p_A.covariance() * F.transpose());

        // define the expected mean lambda function
        // here we use a simple identity function
        auto mean_f = [&] (const VariateA& x) -> VariateA { return f(x); };

        // EXPECT_FALSE(result_gaussian.is_approx(expect_gaussian, eps, true));

        enum { SetSize = Quadrature::template size<VariateA>() };

        auto Z = PointSet<decltype(mean_f(VariateA())), SetSize>();

        quadrature.propergate_gaussian(mean_f, p_A, Z);

        auto mean = Z.center();
        auto Z_c = Z.points();
        auto W = Z.covariance_weights_vector();

        // integrate mean and covariance
        result_gaussian.mean(mean);
        result_gaussian.covariance(Z_c * W.asDiagonal() * Z_c.transpose());

        EXPECT_TRUE(result_gaussian.is_approx(expect_gaussian, eps, true));
    }

    void propergate_gaussian_X_Z()
    {
        using namespace fl;

        auto result_gaussian = GaussianA(DimA);
        auto expect_gaussian = GaussianA(DimA);
        expect_gaussian.mean(f(p_A.mean()));
        expect_gaussian.covariance(F * p_A.covariance() * F.transpose());

        // define the expected mean lambda function
        // here we use a simple identity function
        auto mean_f = [&] (const VariateA& x) -> VariateA { return f(x); };

        // EXPECT_FALSE(result_gaussian.is_approx(expect_gaussian, eps, true));

        enum { SetSize = Quadrature::template size<VariateA>() };

        auto Y = PointSet<VariateA, SetSize>();
        auto Z = PointSet<decltype(mean_f(VariateA())), SetSize>();

        // integrate and check moment results
        quadrature.propergate_gaussian(mean_f, p_A, Y, Z);

        auto mean = Z.center();
        auto Z_c = Z.points();
        auto W = Z.covariance_weights_vector();

        // integrate mean and covariance
        result_gaussian.mean(mean);
        result_gaussian.covariance(Z_c * W.asDiagonal() * Z_c.transpose());

        EXPECT_TRUE(result_gaussian.is_approx(expect_gaussian, eps, true));

        // finally check the Y result whether it actually represents the
        // input Gaussian p_A
        auto temp_gaussian = GaussianA(DimA);

        // center points and get the mean
        auto Y_mean = Y.center();

        // compute point covariacne
        auto Y_c = Y.points();
        auto Y_W = Y.covariance_weights_vector();
        auto Y_cov = Y_c * Y_W.asDiagonal() * Y_c.transpose();

        temp_gaussian.mean(Y_mean);
        temp_gaussian.covariance(Y_cov);
        EXPECT_TRUE(temp_gaussian.is_approx(p_A, eps, true));
    }

    void propergate_gaussian_pxy_Z()
    {
        using namespace fl;

        auto expect_gaussian = GaussianB(DimB);
        auto result_gaussian = GaussianB(DimB);

        expect_gaussian.mean(f(p_A.mean(), p_B.mean()));
        expect_gaussian.covariance(
            H * p_A.covariance() * H.transpose() + p_B.covariance());

        // define the expected mean lambda function
        // here we use a simple identity function
        auto mean_f = [&] (const VariateA& x, const VariateB& y) -> VariateB
        {
            return f(x, y);
        };

        // EXPECT_FALSE(result_gaussian.is_approx(expect_gaussian, eps, true));

        enum { SetSize = Quadrature::template size<VariateA, VariateB>() };

        auto Z = PointSet<decltype(mean_f(VariateA(), VariateB())), SetSize>();

        quadrature.propergate_gaussian(mean_f, p_A, p_B, Z);

        auto mean = Z.center();
        auto Z_c = Z.points();
        auto W = Z.covariance_weights_vector();

        // integrate mean and covariance
        result_gaussian.mean(mean);
        result_gaussian.covariance(Z_c * W.asDiagonal() * Z_c.transpose());

        EXPECT_TRUE(result_gaussian.is_approx(expect_gaussian, eps, true));
    }


    void propergate_gaussian_pxy_X_Y_Z()
    {
        using namespace fl;

        auto expect_gaussian = GaussianB(DimB);
        auto result_gaussian = GaussianB(DimB);

        expect_gaussian.mean(f(p_A.mean(), p_B.mean()));
        expect_gaussian.covariance(
            H * p_A.covariance() * H.transpose() + p_B.covariance());

        // define the expected mean lambda function
        // here we use a simple identity function
        auto mean_f = [&] (const VariateA& x, const VariateB& y) -> VariateB
        {
            return f(x, y);
        };

        // EXPECT_FALSE(result_gaussian.is_approx(expect_gaussian, eps, true));

        enum { SetSize = Quadrature::template size<VariateA, VariateB>() };

        auto X = PointSet<VariateA, SetSize>();
        auto Y = PointSet<VariateB, SetSize>();
        auto Z = PointSet<decltype(mean_f(VariateA(), VariateB())), SetSize>();

        quadrature.propergate_gaussian(mean_f, p_A, p_B, X, Y, Z);

        auto mean = Z.center();
        auto Z_c = Z.points();
        auto W = Z.covariance_weights_vector();

        // integrate mean and covariance
        result_gaussian.mean(mean);
        result_gaussian.covariance(Z_c * W.asDiagonal() * Z_c.transpose());

        EXPECT_TRUE(result_gaussian.is_approx(expect_gaussian, eps, true));

        // finally check the X and Y result whether they actually represents the
        // input Gaussians p_A and p_B, respectively
        auto temp_gaussian_A = GaussianA(DimA);
        auto temp_gaussian_B = GaussianB(DimB);

        // center points and get the mean
        auto X_mean = X.center();
        auto Y_mean = Y.center();

        // compute point covariacne
        auto X_c = X.points();
        auto Y_c = Y.points();

        auto X_W = X.covariance_weights_vector();
        auto Y_W = Y.covariance_weights_vector();

        auto X_cov = X_c * X_W.asDiagonal() * X_c.transpose();
        auto Y_cov = Y_c * Y_W.asDiagonal() * Y_c.transpose();

        temp_gaussian_A.mean(X_mean);
        temp_gaussian_A.covariance(X_cov);
        EXPECT_TRUE(temp_gaussian_A.is_approx(p_A, eps, true));

        temp_gaussian_B.mean(Y_mean);
        temp_gaussian_B.covariance(Y_cov);
        EXPECT_TRUE(temp_gaussian_B.is_approx(p_B, eps, true));
    }

    void integrate_moments_fx_px()
    {
        using namespace fl;

        // compute the expected integration result analytically
        auto expect_gaussian = GaussianA(DimA);
        expect_gaussian.mean(f(p_A.mean()));
        expect_gaussian.covariance(F * p_A.covariance() * F.transpose());

        // create the gaussian which will contail the integration results
        auto result_gaussian = GaussianA(DimA);

        // define the expected mean lambda function
        // here we use a simple identity function
        auto mean_f = [&] (const VariateA& x) -> VariateA { return f(x); };

        // EXPECT_FALSE(result_gaussian.is_approx(expect_gaussian, eps, true));

        // integrate mean and covariance
        auto mean = typename FirstMomentOf<VariateA>::Type();
        auto cov = typename SecondMomentOf<VariateA>::Type();
        quadrature.integrate_moments(mean_f, p_A, mean, cov);

        result_gaussian.mean(mean);
        result_gaussian.covariance(cov);

        EXPECT_TRUE(result_gaussian.is_approx(expect_gaussian, eps, true));
    }

    void integrate_moments_fxy_pxy()
    {
        using namespace fl;

        // compute the expected integration result analytically
        auto expect_gaussian = GaussianB(DimB);

        expect_gaussian.mean(f(p_A.mean(), p_B.mean()));
        expect_gaussian.covariance(
            H * p_A.covariance() * H.transpose() + p_B.covariance());

        // define the expected mean lambda function
        // here we use a simple identity function
        auto mean_f = [&] (const VariateA& x, const VariateB& y) -> VariateB
        {
            return f(x, y);
        };

        // integrate mean and covariance
        auto mean = typename FirstMomentOf<VariateB>::Type();
        auto cov = typename SecondMomentOf<VariateB>::Type();
        quadrature.integrate_moments(mean_f, p_A, p_B, mean, cov);

        // create the gaussian which will contail the integration results
        auto result_gaussian = GaussianB(DimB);

        // EXPECT_FALSE(result_gaussian.is_approx(expect_gaussian, eps, true));

        result_gaussian.mean(mean);
        result_gaussian.covariance(cov);

        EXPECT_TRUE(result_gaussian.is_approx(expect_gaussian, eps, true));
    }

protected:
    /* parameter of the linear function f(x) = A*x */
    MatrixAA F;
    MatrixAB H;

    /* source Gaussian distributions used within the integrals */
    GaussianA p_A;
    GaussianB p_B;

    /* our integrator */
    Quadrature quadrature;

    fl::Real eps;
};

TYPED_TEST_CASE_P(SigmaPointQuadratureTests);

TYPED_TEST_P(SigmaPointQuadratureTests, integrate_fx_px)
{
    TestFixture::integrate_fx_px();
}

TYPED_TEST_P(SigmaPointQuadratureTests, integrate_fxy_pxy)
{
    TestFixture::integrate_fxy_pxy();
}

TYPED_TEST_P(SigmaPointQuadratureTests, propergate_gaussian_Z)
{
    TestFixture::propergate_gaussian_Z();
}

TYPED_TEST_P(SigmaPointQuadratureTests, propergate_gaussian_X_Z)
{
    TestFixture::propergate_gaussian_X_Z();
}

TYPED_TEST_P(SigmaPointQuadratureTests, propergate_gaussian_pxy_Z)
{
    TestFixture::propergate_gaussian_pxy_Z();
}

TYPED_TEST_P(SigmaPointQuadratureTests, propergate_gaussian_pxy_X_Y_Z)
{
    TestFixture::propergate_gaussian_pxy_X_Y_Z();
}

TYPED_TEST_P(SigmaPointQuadratureTests, integrate_moments_fx_px)
{
    TestFixture::integrate_moments_fx_px();
}

TYPED_TEST_P(SigmaPointQuadratureTests, integrate_moments_fxy_pxy)
{
    TestFixture::integrate_moments_fxy_pxy();
}

REGISTER_TYPED_TEST_CASE_P(SigmaPointQuadratureTests,
                           integrate_fx_px,
                           integrate_fxy_pxy,
                           propergate_gaussian_Z,
                           propergate_gaussian_X_Z,
                           propergate_gaussian_pxy_Z,
                           propergate_gaussian_pxy_X_Y_Z,
                           integrate_moments_fx_px,
                           integrate_moments_fxy_pxy);

namespace internal
{

// Transform configuration selection helper
template <typename T> struct TransformSelection;

// TransformSelection for MonteCarlo integration
template <typename PSP>
struct TransformSelection<fl::MonteCarloTransform<PSP>>
{
    static constexpr fl::Real epsilon = fl::Real(0.5);
    typedef fl::MonteCarloTransform<PSP> Transform;
};

// TransformSelection for deterministic Unscented integration
template <> struct TransformSelection<fl::UnscentedTransform>
{
    static constexpr fl::Real epsilon = fl::Real(1.e-9);
    typedef fl::UnscentedTransform Transform;
};

}

template <int DimensionA, int DimensionB, typename Transform>
struct TestConfiguration
{
    enum : signed int
    {
        DimA = DimensionA,
        DimB = DimensionB
    };

    typedef internal::TransformSelection<Transform> TransformSelection;
};

