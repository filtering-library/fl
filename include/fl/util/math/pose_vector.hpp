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
 * \file orientation_state_transition_model.hpp
 * \date July 2015
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#pragma once



#include <Eigen/Dense>
#include <fl/util/types.hpp>

#include <fl/util/math/euler_vector.hpp>

namespace fl
{


/// basic functionality for both vectors and blocks ****************************
template <typename Base>
class PoseBase: public Base
{
public:
    enum
    {
        BLOCK_SIZE = 3,
        POSITION_INDEX = 0,
        EULER_VECTOR_INDEX = 3
    };

    // types *******************************************************************
    typedef Eigen::Matrix<Real, 3, 1>              Vector;
    typedef Eigen::Matrix<Real, 4, 4>              HomogeneousMatrix;
    typedef typename Eigen::Transform<Real, 3, Eigen::Affine> Affine;

    typedef Eigen::VectorBlock<Base, BLOCK_SIZE>   PositionBlock;
    typedef EulerBlock<Base>                       OrientationBlock;

    typedef PoseBase<Eigen::Matrix<Real, 6, 1>>     PoseVector;

    // constructor and destructor **********************************************
    PoseBase(const Base& vector): Base(vector) { }
    virtual ~PoseBase() noexcept {}

    // operators ***************************************************************
    template <typename T>
    void operator = (const Eigen::MatrixBase<T>& vector)
    {
        *((Base*)(this)) = vector;
    }

    // accessors ***************************************************************
    virtual Vector position() const
    {
        return this->template middleRows<BLOCK_SIZE>(POSITION_INDEX);
    }
    virtual EulerVector orientation() const
    {
        return this->template middleRows<BLOCK_SIZE>(EULER_VECTOR_INDEX);
    }
    virtual HomogeneousMatrix homogeneous() const
    {
        HomogeneousMatrix H(HomogeneousMatrix::Identity());
        H.topLeftCorner(3, 3) = orientation().rotation_matrix();
        H.topRightCorner(3, 1) = position();

        return H;
    }
    virtual Affine affine() const
    {
        Affine A;
        A.linear() = orientation().rotation_matrix();
        A.translation() = position();

        return A;
    }
    virtual PoseVector inverse() const
    {
        PoseVector inv(PoseVector::Zero());
        inv.homogeneous(this->homogeneous().inverse());
        return inv;
    }

    // mutators ****************************************************************
    PositionBlock position()
    {
      return PositionBlock(*this, POSITION_INDEX);
    }
    OrientationBlock orientation()
    {
      return OrientationBlock(*this, EULER_VECTOR_INDEX);
    }
    virtual void homogeneous(const HomogeneousMatrix& H)
    {
        orientation().rotation_matrix(H.topLeftCorner(3, 3));
        position() = H.topRightCorner(3, 1);
    }
    virtual void affine(const Affine& A)
    {
       orientation().rotation_matrix(A.rotation());
       position() = A.translation();
    }
    virtual void set_zero_pose()
    {
        this->setZero();
    }
    // operators ***************************************************************
    template <typename T>
    PoseVector operator * (const PoseBase<T>& factor)
    {
        PoseVector product(PoseVector::Zero());
        product.homogeneous(
                    this->homogeneous() * factor.homogeneous());
        return product;
    }
};


/// implementation for vectors *************************************************
class PoseVector: public PoseBase<Eigen::Matrix<Real, 6, 1>>
{
public:
    typedef PoseBase<Eigen::Matrix<Real, 6, 1>> Base;

    // constructor and destructor **********************************************
    PoseVector(): Base(Base::Zero()) { }

    template <typename T>
    PoseVector(const Eigen::MatrixBase<T>& vector): Base(vector) { }

    virtual ~PoseVector() noexcept {}
};


/// implementation for blocks **************************************************
template <typename Vector>
class PoseBlock: public PoseBase<Eigen::VectorBlock<Vector, 6>>
{
public:
    typedef Eigen::VectorBlock<Vector, 6>   Block;
    typedef PoseBase<Block>                 Base;

    using Base::operator=;

    // constructor and destructor **********************************************
    PoseBlock(const Block& block): Base(block) { }
    PoseBlock(Vector& vector, int start): Base(Block(vector, start)) { }
    virtual ~PoseBlock() noexcept {}
};


}


