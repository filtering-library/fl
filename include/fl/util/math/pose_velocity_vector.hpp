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
 * \date July 2015
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
        BLOCK_SIZE = 3,
        POSITION_INDEX = 0,
        POSE_INDEX = 0,
        EULER_VECTOR_INDEX = 3,
        LINEAR_VELOCITY_INDEX = 6,
        ANGULAR_VELOCITY_INDEX = 9
    };

    // types *******************************************************************
    typedef Eigen::Matrix<Real, 3, 1>              Vector;
    typedef Eigen::Matrix<Real, 4, 4>              HomogeneousMatrix;
    typedef Eigen::VectorBlock<Base, BLOCK_SIZE>   PositionBlock;
    typedef EulerBlock<Base>                       OrientationBlock;

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
        return this->template middleRows<BLOCK_SIZE>(POSE_INDEX);
    }
    virtual Vector linear_velocity() const
    {
        return this->template middleRows<BLOCK_SIZE>(LINEAR_VELOCITY_INDEX);
    }
    virtual Vector angular_velocity() const
    {
        return this->template middleRows<BLOCK_SIZE>(ANGULAR_VELOCITY_INDEX);
    }

    virtual HomogeneousMatrix homogeneous_matrix() const
    {
        HomogeneousMatrix H(HomogeneousMatrix::Identity());
        H.topLeftCorner(3, 3) = euler_vector().rotation_matrix();
        H.topRightCorner(3, 1) = position();

        return H;
    }

    // mutators ****************************************************************
    PositionBlock position()
    {
      return PositionBlock(this->derived(), POSITION_INDEX);
    }
    OrientationBlock euler_vector()
    {
      return OrientationBlock(this->derived(), EULER_VECTOR_INDEX);
    }
    virtual void homogeneous_matrix(const HomogeneousMatrix& H)
    {
        euler_vector().rotation_matrix(H.topLeftCorner(3, 3));
        position() = H.topRightCorner(3, 1);
    }
};

















class PoseVelocityVector: public Eigen::Matrix<Real, 12, 1>
{
public:
    enum
    {
        BLOCK_SIZE = 3,
        POSITION_INDEX = 0,
        EULER_VECTOR_INDEX = 3,
        LINEAR_VELOCITY_INDEX = 6,
        ANGULAR_VELOCITY_INDEX = 9
    };

    typedef Eigen::Matrix<Real, 12, 1>       StateVector;
    typedef Eigen::Matrix<Real, 3, 1>               Vector;



    // rotation types
    typedef Eigen::AngleAxis<Real>    AngleAxis;
    typedef Eigen::Quaternion<Real>   Quaternion;
    typedef Eigen::Matrix<Real, 3, 3> RotationMatrix;
    typedef Eigen::Matrix<Real, 4, 4> HomogeneousMatrix;
    typedef typename Eigen::Transform<Scalar, 3, Eigen::Affine> Affine;

    typedef Eigen::VectorBlock<StateVector, BLOCK_SIZE>   StateBlock;
    typedef EulerBlock<StateVector>                       EulerStateBlock;


    // constructor and destructor
    PoseVelocityVector() { }
    template <typename T> RigidBodyState(const Eigen::MatrixBase<T>& state_vector)
    {
        *this = state_vector;
    }
    virtual ~PoseVelocityVector() {}

    template <typename T>
    void operator = (const Eigen::MatrixBase<T>& state_vector)
    {
        *((StateVector*)(this)) = state_vector;
    }
  

    /// accessors **************************************************************
    virtual Vector position() const
    {
        return this->middleRows<BLOCK_SIZE>(POSITION_INDEX);
    }
    virtual EulerVector euler_vector() const
    {
        return this->middleRows<BLOCK_SIZE>(EULER_VECTOR_INDEX);
    }
    virtual Vector linear_velocity() const
    {
        return this->middleRows<BLOCK_SIZE>(LINEAR_VELOCITY_INDEX);
    }
    virtual Vector angular_velocity() const
    {
        return this->middleRows<BLOCK_SIZE>(ANGULAR_VELOCITY_INDEX);
    }

    /// mutators ***************************************************************
    StateBlock position()
    {
      return StateBlock(this->derived(), POSITION_INDEX);
    }
    EulerStateBlock euler_vector()
    {
      return EulerStateBlock(this->derived(), EULER_VECTOR_INDEX);
    }
    StateBlock linear_velocity()
    {
      return StateBlock(this->derived(), LINEAR_VELOCITY_INDEX);
    }
    StateBlock angular_velocity()
    {
      return StateBlock(this->derived(), ANGULAR_VELOCITY_INDEX);
    }


//    virtual void pose(const Affine& affine, )
//    {
//       quaternion(Quaternion(affine.rotation()), body_index);
//       position(body_index) = affine.translation();
//    }


//    virtual HomogeneousMatrix homogeneous_matrix(const size_t& object_index = 0) const
//    {
//        HomogeneousMatrix homogeneous_matrix(HomogeneousMatrix::Identity());
//        homogeneous_matrix.topLeftCorner(3, 3) = rotation_matrix(object_index);
//        homogeneous_matrix.topRightCorner(3, 1) = position(object_index);

//        return homogeneous_matrix;
//    }

};

}

#endif
