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
 * \file scalar_matrix_test.hpp
 * \date August 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>

#include <fl/util/scalar_matrix.hpp>

#include <boost/math/distributions/chi_squared.hpp>


/* == creation and conversion =============================================== */

TEST(ScalarMatrixTests, create)
{
    fl::ScalarMatrix s;
    EXPECT_EQ(s(0), 0);
}

TEST(ScalarMatrixTests, create_with_value)
{
    fl::ScalarMatrix s(42.);
    EXPECT_DOUBLE_EQ(s(0), 42.);
}

TEST(ScalarMatrixTests, assign_scalar_to_ScalarMatrix)
{
    fl::ScalarMatrix s;
    s = 43.;
    EXPECT_DOUBLE_EQ(s(0), 43.);
}

TEST(ScalarMatrixTests, assign_ScalarMatrix_to_scalar)
{
    fl::Real r = 0;
    fl::ScalarMatrix s(44);
    r = s;
    EXPECT_DOUBLE_EQ(s(0), r);
}

TEST(ScalarMatrixTests, create_with_value_implicit_conversion_test)
{
    fl::ScalarMatrix s(42.);
    EXPECT_DOUBLE_EQ(s, 42.);
}

/* == addition (+) ========================================================== */

TEST(ScalarMatrixTests, plus_scalar_value)
{
    fl::ScalarMatrix s(42);
    s += 43.;
    EXPECT_DOUBLE_EQ(s, 42. + 43.);
}

TEST(ScalarMatrixTests, plus_scalar_matrix)
{
    fl::ScalarMatrix s(42);
    s += fl::ScalarMatrix (43.);
    EXPECT_DOUBLE_EQ(s, 42. + 43.);
}

TEST(ScalarMatrixTests, scalar_matrix_plus_scalar_matrix)
{
    auto a = fl::ScalarMatrix (42);
    auto b = fl::ScalarMatrix (43);
    fl::ScalarMatrix s;

    s = a + b;
    EXPECT_DOUBLE_EQ(s, 42. + 43.);
}

TEST(ScalarMatrixTests, scalar_plus_scalar_matrix)
{
    fl::Real a = 42;
    auto b = fl::ScalarMatrix (43);
    fl::ScalarMatrix s;

    s = a + b;
    EXPECT_DOUBLE_EQ(s, 42. + 43.);

    s = b + a;
    EXPECT_DOUBLE_EQ(s, 42. + 43.);
}

/* == subtraction (-) ======================================================= */

TEST(ScalarMatrixTests, minus_scalar_value)
{
    fl::ScalarMatrix s(42);
    s -= 43.;
    EXPECT_DOUBLE_EQ(s, 42. - 43.);
}

TEST(ScalarMatrixTests, minus_scalar_matrix)
{
    fl::ScalarMatrix s(42);
    s -= fl::ScalarMatrix (43.);
    EXPECT_DOUBLE_EQ(s, 42. - 43.);
}

TEST(ScalarMatrixTests, scalar_matrix_minus_scalar_matrix)
{
    auto a = fl::ScalarMatrix (42);
    auto b = fl::ScalarMatrix (43);
    fl::ScalarMatrix s;

    s = a - b;
    EXPECT_DOUBLE_EQ(s, 42. - 43.);
}

TEST(ScalarMatrixTests, scalar_minus_scalar_matrix)
{
    fl::Real a = 42;
    auto b = fl::ScalarMatrix (43);
    fl::ScalarMatrix s;

    s = a - b;
    EXPECT_DOUBLE_EQ(s, 42. - 43.);

    s = b - a;
    EXPECT_DOUBLE_EQ(s, 43. - 42.);
}

/* == multiplication (*) ==================================================== */

TEST(ScalarMatrixTests, multiply_scalar_value)
{
    fl::ScalarMatrix s(42);
    s *= 43.;
    EXPECT_DOUBLE_EQ(s, 42. * 43.);
}

TEST(ScalarMatrixTests, multiply_scalar_matrix)
{
    fl::ScalarMatrix s(42);
    s *= fl::ScalarMatrix (43.);
    EXPECT_DOUBLE_EQ(s, 42. * 43.);
}

TEST(ScalarMatrixTests, scalar_matrix_multiply_scalar_matrix)
{
    auto a = fl::ScalarMatrix (42);
    auto b = fl::ScalarMatrix (43);
    fl::ScalarMatrix s;

    s = a * b;
    EXPECT_DOUBLE_EQ(s, 42. * 43.);
}

TEST(ScalarMatrixTests, scalar_multiply_scalar_matrix)
{
    fl::Real a = 42;
    auto b = fl::ScalarMatrix (43);
    fl::ScalarMatrix s;

    s = a * b;
    EXPECT_DOUBLE_EQ(s, 42. * 43.);

    s = b * a;
    EXPECT_DOUBLE_EQ(s, 43. * 42.);
}

/* == division (/) ========================================================== */

TEST(ScalarMatrixTests, divide_scalar_value)
{
    fl::ScalarMatrix s(42);
    s /= 43.;
    EXPECT_DOUBLE_EQ(s, 42. / 43.);
}

TEST(ScalarMatrixTests, divide_scalar_matrix)
{
    fl::ScalarMatrix s(42);
    s /= fl::ScalarMatrix (43.);
    EXPECT_DOUBLE_EQ(s, 42. / 43.);
}

TEST(ScalarMatrixTests, scalar_matrix_divide_scalar_matrix)
{
    auto a = fl::ScalarMatrix (42);
    auto b = fl::ScalarMatrix (43);
    fl::ScalarMatrix s;

    s = a / b;
    EXPECT_DOUBLE_EQ(s, 42. / 43.);
}

TEST(ScalarMatrixTests, scalar_divide_scalar_matrix)
{
    fl::Real a = 42;
    auto b = fl::ScalarMatrix (43);
    fl::ScalarMatrix s;

    s = a / b;
    EXPECT_DOUBLE_EQ(s, 42. / 43.);

    s = b / a;
    EXPECT_DOUBLE_EQ(s, 43. / 42.);
}

/* == ++ and -- ============================================================= */

TEST(ScalarMatrixTests, prefix_increment)
{
    fl::ScalarMatrix s(42);
    EXPECT_DOUBLE_EQ(++s, 42 + 1);
    EXPECT_DOUBLE_EQ(s, 42 + 1);
}

TEST(ScalarMatrixTests, postfix_increment)
{
    fl::ScalarMatrix s(42);
    EXPECT_DOUBLE_EQ(s++, 42);
    EXPECT_DOUBLE_EQ(s, 42 + 1);
}

TEST(ScalarMatrixTests, prefix_decrement)
{
    fl::ScalarMatrix s(42);
    EXPECT_DOUBLE_EQ(--s, 42 - 1);
    EXPECT_DOUBLE_EQ(s, 42 - 1);
}

TEST(ScalarMatrixTests, postfix_decrement)
{
    fl::ScalarMatrix s(42);
    EXPECT_DOUBLE_EQ(s--, 42);
    EXPECT_DOUBLE_EQ(s, 42 - 1);
}
