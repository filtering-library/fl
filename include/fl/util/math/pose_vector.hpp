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

#ifndef FL__UTIL__MATH__POSE_VECTOR_HPP
#define FL__UTIL__MATH__POSE_VECTOR_HPP


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
    typedef Eigen::VectorBlock<Base, BLOCK_SIZE>   StateBlock;
    typedef EulerBlock<Base>                       EulerStateBlock;

    // constructor and destructor **********************************************
    PoseBase(const Base& vector): Base(vector) { }
    virtual ~PoseBase() {}

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
    virtual EulerVector euler_vector() const
    {
        return this->template middleRows<BLOCK_SIZE>(EULER_VECTOR_INDEX);
    }
    virtual HomogeneousMatrix homogeneous_matrix() const
    {
        HomogeneousMatrix H(HomogeneousMatrix::Identity());
        H.topLeftCorner(3, 3) = euler_vector().rotation_matrix();
        H.topRightCorner(3, 1) = position();

        return H;
    }

    // mutators ****************************************************************
    StateBlock position()
    {
      return StateBlock(this->derived(), POSITION_INDEX);
    }
    EulerStateBlock euler_vector()
    {
      return EulerStateBlock(this->derived(), EULER_VECTOR_INDEX);
    }
    virtual void homogeneous_matrix(const HomogeneousMatrix& H)
    {
        euler_vector().rotation_matrix(H.topLeftCorner(3, 3));
        position() = H.topRightCorner(3, 1);
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

    virtual ~PoseVector() {}

    // operators ***************************************************************
    PoseVector operator * (const PoseVector& factor)
    {
        PoseVector product;
        product.homogeneous_matrix(
                    this->homogeneous_matrix() * factor.homogeneous_matrix());
        return product;
    }

    // accessor ****************************************************************
    virtual PoseVector inverse() const
    {
        return - (*this);
    }
};


/// implementation for blocks **************************************************
template <typename Vector>
class PoseBlock: public PoseBase<Eigen::VectorBlock<Vector, 6>>
{
public:
    typedef Eigen::VectorBlock<Vector, 6>   Block;
    typedef PoseBase<Block>                 Base;

    // constructor and destructor **********************************************
    PoseBlock(const Block& block): Base(block) { }
    PoseBlock(Vector& vector, int start): Base(Block(vector, start)) { }
    virtual ~PoseBlock() {}
};


}

#endif
