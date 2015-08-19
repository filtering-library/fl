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
 * \file traits.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__UTIL__TRAITS_HPP
#define FL__UTIL__TRAITS_HPP

#include <Eigen/Dense>

#include "types.hpp"

namespace fl
{

/**
 * \ingroup traits
 * \def from_traits
 * \brief Helper macro to import typedefs from the traits of a class
 *
 * typedef from_traits(SomeTypeName);
 *
 * is the short hand for
 *
 * typedef typename Traits<This>::SomeTypeName SomeTypeName;
 *
 * \note from_this(.) requires the typedef \c This of the current class
 */
#define from_traits(TypeName) typename Traits<This>::TypeName TypeName

#if defined(__GXX_EXPERIMENTAL_CXX0X__)
    #define override
#endif

/**
 * \ingroup traits
 * \brief Generic trait template
 *
 * Filters, models and distributions may specify a \c Traits specialization
 */
template <typename> struct Traits { };

/**
 * \ingroup traits
 *
 * \brief \c IsDynamic<int> trait for static dynamic-size checks.
 *
 * Generic IsDynamic<int> definition which evaluates to false.
 *
 * Examples
 *
 * static_assert(IsDynamic<MyEigenMatrixType::SizeAtCompileTime>(), "");
 *
 * if (IsDynamic<MyEigenMatrixType::SizeAtCompileTime>::value) ...
 */
template <int Size> struct IsDynamic
{
    static_assert(Size > Eigen::Dynamic, "Invalid static size");

    static constexpr bool value = false;
    constexpr operator bool () { return value; }
};

/**
 * \ingroup traits
 * \brief \c IsDynamic<-1> or \c IsDynamic<Eigen::Dynamic> trait for static
 * dynamic-size checks.
 *
 * A specialization IsDynamic<Eigen::Dynamic> definition which evaluates to
 * true.
 *
 * Examples
 *
 * // overloaded operator bool ()
 * static_assert(!IsDynamic<MyEigenMatrixType::SizeAtCompileTime>(), "");
 *
 * if (!IsDynamic<MyEigenMatrixType::SizeAtCompileTime>::value) ...
 */
template <> struct IsDynamic<Eigen::Dynamic>
{
    static constexpr bool value = true;
    constexpr operator bool () { return value; }
};

/**
 * \ingroup traits
 *
 * \brief \c IsFixed<int> trait for static fixed-size checks.
 *
 * Generic IsFixed<int> definition which evaluates to true. This traits is the
 * opposite of IsDynamic.
 *
 * Examples
 *
 * static_assert(IsFixed<MyEigenMatrixType::SizeAtCompileTime>::value, "");
 *
 * // overloaded operator bool ()
 * if (IsFixed<MyEigenMatrixType::SizeAtCompileTime>()) ...
 */
template <int Size> struct IsFixed
{
    static_assert(Size > Eigen::Dynamic, "Invalid static size");

    static constexpr bool value = true;
    constexpr operator bool () { return value; }
};

/**
 * \ingroup traits
 * \brief \c IsFixed<-1> or \c IsFixed<Eigen::Dynamic> trait for static
 * fixed-size checks.
 *
 * A specialization IsFixed<Eigen::Dynamic> definition which evaluates to
 * false. This traits is the opposite of IsDynamic<Eigen::Dynamic>.
 *
 * Examples
 *
 * static_assert(!IsFixed<MyEigenMatrixType::SizeAtCompileTime>(), "");
 *
 * if (!IsFixed<MyEigenMatrixType::SizeAtCompileTime>::value) ...
 */
template <> struct IsFixed<Eigen::Dynamic>
{
    static constexpr bool value = false;
    constexpr operator bool () { return value; }
};

/**
 * \ingroup traits
 *
 * \brief Mapps Eigen::Dynamic onto 0.
 *
 * For any type matrix or column vector the dimension is the number of rows,
 * i.e. Matrix::RowsAtCompileTime. If the Matrix::SizeAtCompileTime enum is not
 * equal Eigen::Dynamic (-1) the dimension is set to Matrix::RowsAtCompileTime.
 * Otherwise, 0 is returned.
 *
 * Examples
 *
 * static_assert(DimensionOf<MyEigenMatrixType>() > 0, "Dim must be greater 0");
 *
 * static_assert(DimensionOf<MyEigenMatrixType>::value > 0, "Dim must be .. 0");
 *
 * Eigen::VectorXd vector(DimensionOf<MyEigenMatrixType>());
 */
template <typename Matrix> struct DimensionOf
{
    enum : signed int { Value = IsFixed<Matrix::SizeAtCompileTime>()
                                    ? Matrix::RowsAtCompileTime
                                    : 0 };

    constexpr operator int () { return Value; }
};

/**
 * \ingroup traits
 * \brief Returns the compile time size of a matrix or vector.
 *
 * SizeOf<M>() is shorthand for M::SizeAtCompileTime
 */
template <typename Matrix> struct SizeOf
{
    enum : signed int { Value = Matrix::SizeAtCompileTime };

    constexpr operator int () { return Value; }
};

/**
 * \ingroup traits
 *
 */
template <int Dimension> struct ToDimension
{
    enum : signed int { Value = Dimension == Eigen::Dynamic ? 0 : Dimension };

    constexpr operator int () { return Value; }
};


/**
 * \ingroup traits
 *
 * \brief Returns simple the max integer of A and B
 */
template <int A, int B> struct MaxOf
{
    enum : signed int { Value = (A > B) ? A : B };
    static constexpr int value = (A > B) ? A : B;
    constexpr operator int () { return Value; }
};

/**
 * \ingroup traits
 *
 * \brief Returns simple the min integer of A and B
 */
template <int A, int B> struct MinOf
{
    enum : signed int { Value = (A < B) ? A : B };
    static constexpr int value = (A < B) ? A : B;
    constexpr operator int () { return Value; }
};

/**
 * \ingroup traits
 *
 * Defines the second moment type of a given variate
 */
template <typename Variate>
struct SecondMomentOf
{
    enum: signed int { Dimension = SizeOf<Variate>::Value };
    typedef Eigen::Matrix<Real, Dimension, Dimension> Type;
};

/**
 * \ingroup traits
 */
template <typename Variate>
struct FirstMomentOf
{
    enum: signed int { Dimension = SizeOf<Variate>::Value };
    typedef Eigen::Matrix<Real, Dimension, 1> Type;
};

/**
 * \ingroup traits
 *
 * Defines the second moment type of a given variate
 */
template <typename Variate>
struct DiagonalSecondMomentOf
{
    enum: signed int { Dimension = SizeOf<Variate>::Value };
    typedef Eigen::DiagonalMatrix<Real, Dimension> Type;
};

/**
 * \ingroup traits
 *
 * Defines the second moment type of a given variate
 */
template <>
struct SecondMomentOf<Real>
{
    typedef Real Type;
};

/**
 * \ingroup traits
 */
template <>
struct FirstMomentOf<Real>
{
    typedef Real Type;
};


/**
 * \ingroup traits
 * Defines a Real variate of the specified size
 */
template <int Size>
struct VariateOfSize
{
    typedef Eigen::Matrix<Real, Size, 1> Type;
};


/**
 * \ingroup traits
 */
template <typename Model> struct IsAdditive
{
    enum: bool
    {
        Value = std::is_base_of<internal::AdditiveNoiseModelType, Model>::value
    };
};

/**
 * \ingroup traits
 */
template <typename Model> struct IsAdditiveUncorrelated
{
    enum: bool
    {
        Value = std::is_base_of<
                    internal::AdditiveUncorrelatedNoiseModelType,
                    Model
                >::value
    };
};

/**
 * \ingroup traits
 */
template <typename Model> struct IsNonAdditive
{
    enum: bool
    {
        Value = std::is_base_of<internal::NonAdditiveNoiseModelType, Model>::value
    };
};

/**
 * \internal
 * \ingroup traits
 */
template <typename Model, typename ...> struct AdditivityOf;

/**
 * \internal
 * \ingroup traits
 */
template <typename Model>
struct AdditivityOf<Model, internal::AdditiveNoiseModelType>
{
    typedef Additive<Model> Type;
};

/**
 * \internal
 * \ingroup traits
 */
template <typename Model>
struct AdditivityOf<Model, internal::AdditiveUncorrelatedNoiseModelType>
{
    typedef AdditiveUncorrelated<Model> Type;
};

/**
 * \internal
 * \ingroup traits
 */
template <typename Model>
struct AdditivityOf<Model, internal::NonAdditiveNoiseModelType>
{
    typedef NonAdditive<Model> Type;
};

/**
 * \internal
 * \ingroup traits
 */
template <typename Model>
struct AdditivityOf<Additive<Model>>
{
    typedef Additive<Model> Type;
};

/**
 * \internal
 * \ingroup traits
 */
template <typename Model>
struct AdditivityOf<AdditiveUncorrelated<Model>>
{
    typedef AdditiveUncorrelated<Model> Type;
};

/**
 * \internal
 * \ingroup traits
 */
template <typename Model>
struct AdditivityOf<NonAdditive<Model>>
{
    typedef NonAdditive<Model> Type;
};

/**
 * \ingroup traits
 * \brief Provides access to the type of the model
 *
 * The model type will be one of the following
 *  - \c NonAdditive (Noise)
 *  - \c Additive (Noise)
 *  - \c AdditiveUncorrelated (Noise)
 */
template <typename Model> struct AdditivityOf<Model>
{
    typedef typename AdditivityOf<Model, typename Model::Type>::Type Type;
};

/**
 * \internal
 * \ingroup traits
 */
template <typename Model> struct RemoveAdditivityOf
{
    typedef Model Type;
};

template <typename Model>
struct RemoveAdditivityOf<Additive<Model>>
{
    typedef Model Type;
};

template <typename Model>
struct RemoveAdditivityOf<AdditiveUncorrelated<Model>>
{
    typedef Model Type;
};

template <typename Model>
struct RemoveAdditivityOf<NonAdditive<Model>>
{
    typedef Model Type;
};

/**
 * \internal
 *
 * \ingroup traits
 */
template <typename Model>
struct ForwardLinearModelOnly
{
    static_assert(std::is_base_of<internal::LinearModelType, Model>::value,
                  "The specified model is not linear!");

    typedef Model Type;
};

}

#endif

