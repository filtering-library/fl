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
 * \date August 2015
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */


#ifndef FL__UTIL__MATH__POSE_VELOCITY_VECTOR_HPP
#define FL__UTIL__MATH__POSE_VELOCITY_VECTOR_HPP

#include <Eigen/Dense>
#include <fl/util/types.hpp>
#include <fl/util/math/euler_vector.hpp>
#include <fl/util/math/pose_vector.hpp>

namespace fl
{



/// basic functionality for both vectors and blocks ****************************
template <typename Base>
class PoseVelocityBase: public Base
{
public:
    enum
    {
        POSE_INDEX = 0,
        LINEAR_VELOCITY_INDEX = 6,
        ANGULAR_VELOCITY_INDEX = 9,

        POSE_SIZE = 6,
        VELOCITY_SIZE = 3,
    };

    // types *******************************************************************
    typedef Eigen::Matrix<Real, VELOCITY_SIZE, 1>       VelocityVector;
    typedef Eigen::VectorBlock<Base, VELOCITY_SIZE>     VelocityBlock;

    // constructor and destructor **********************************************
    PoseVelocityBase(const Base& vector): Base(vector) { }
    virtual ~PoseVelocityBase() {}

    // operators ***************************************************************
    template <typename T>
    void operator = (const Eigen::MatrixBase<T>& vector)
    {
        *((Base*)(this)) = vector;
    }

    // accessors ***************************************************************
    virtual PoseVector pose() const
    {
        return this->template middleRows<POSE_SIZE>(POSE_INDEX);
    }
    virtual VelocityVector linear_velocity() const
    {
        return this->template middleRows<VELOCITY_SIZE>(LINEAR_VELOCITY_INDEX);
    }
    virtual VelocityVector angular_velocity() const
    {
        return this->template middleRows<VELOCITY_SIZE>(ANGULAR_VELOCITY_INDEX);
    }

    // mutators ****************************************************************
    PoseBlock<Base> pose()
    {
        return PoseBlock<Base>(*this, POSE_INDEX);
    }
    VelocityBlock linear_velocity()
    {
        return VelocityBlock(*this, LINEAR_VELOCITY_INDEX);
    }
    VelocityBlock angular_velocity()
    {
        return VelocityBlock(*this, ANGULAR_VELOCITY_INDEX);
    }
};



/// implementation for vectors *************************************************
class PoseVelocityVector: public PoseVelocityBase<Eigen::Matrix<Real, 12, 1>>
{
public:
    typedef PoseVelocityBase<Eigen::Matrix<Real, 12, 1>> Base;

    // constructor and destructor **********************************************
    PoseVelocityVector(): Base(Base::Zero()) { }

    template <typename T>
    PoseVelocityVector(const Eigen::MatrixBase<T>& vector): Base(vector) { }

    virtual ~PoseVelocityVector() {}
};


/// implementation for blocks **************************************************
template <typename Vector>
class PoseVelocityBlock: public PoseVelocityBase<Eigen::VectorBlock<Vector, 12>>
{
public:
    typedef Eigen::VectorBlock<Vector, 12>   Block;
    typedef PoseVelocityBase<Block>          Base;

    using Base::operator=;

    // constructor and destructor **********************************************
    PoseVelocityBlock(const Block& block): Base(block) { }
    PoseVelocityBlock(Vector& vector, int start): Base(Block(vector, start)) { }
    virtual ~PoseVelocityBlock() {}
};



}

#endif
