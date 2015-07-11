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

#ifndef FL__UTIL__MATH__EULER_VECTOR_HPP
#define FL__UTIL__MATH__EULER_VECTOR_HPP


#include <Eigen/Dense>
#include <fl/util/types.hpp>

namespace fl
{

/// basic functionality for both vectors and blocks ****************************
template <typename Base>
class EulerBase: public Base
{
public:
    // types *******************************************************************
    typedef typename Base::Scalar           Scalar;
    typedef Eigen::Matrix<Scalar, 3, 1>     Axis;
    typedef Eigen::AngleAxis<Scalar>        AngleAxis;
    typedef Eigen::Quaternion<Scalar>       Quaternion;
    typedef Eigen::Matrix<Scalar, 3, 3>     RotationMatrix;

    // constructor and destructor **********************************************
    EulerBase(const Base& vector): Base(vector) { }
    virtual ~EulerBase() {}

    // accessor ****************************************************************
    virtual Scalar angle() const
    {
        return this->norm();
    }
    virtual Axis axis() const
    {
        Axis  ax =  this->normalized();

        if(!std::isfinite(ax.sum()))
        {
            return Axis(1,0,0);
        }
        return ax;
    }
    virtual AngleAxis angle_axis() const
    {
        return AngleAxis(angle(), axis());
    }
    virtual Quaternion quaternion() const
    {
        return Quaternion(angle_axis());
    }
    virtual RotationMatrix rotation_matrix() const
    {
        return RotationMatrix(quaternion());
    }

    // mutators ****************************************************************
    virtual void angle_axis(const Scalar& angle, const Axis& axis)
    {
        *((Base*)(this)) = angle * axis;
    }
    virtual void angle_axis(const AngleAxis& ang_axis)
    {
        angle_axis(ang_axis.angle(), ang_axis.axis());
    }
    virtual void quaternion(const Quaternion& quat)
    {
        angle_axis(AngleAxis(quat));
    }
    virtual void rotation_matrix(const RotationMatrix& rot_mat)
    {
        angle_axis(AngleAxis(rot_mat));
    }
};


/// implementation for vectors *************************************************
class EulerVector: public EulerBase<Eigen::Matrix<Real, 3, 1>>
{
public:
    typedef EulerBase<Eigen::Matrix<Real, 3, 1>> Base;

    // constructor and destructor **********************************************
    EulerVector(): Base(Base::Zero()) { }

    template <typename T>
    EulerVector(const Eigen::MatrixBase<T>& vector): Base(vector) { }

    virtual ~EulerVector() {}

    // operators ***************************************************************
    EulerVector operator * (const EulerVector& factor)
    {
        EulerVector product;
        product.quaternion(this->quaternion() * factor.quaternion());
        return product;
    }

    // accessor ****************************************************************
    virtual EulerVector inverse() const
    {
        return - (*this);
    }
};


/// implementation for blocks **************************************************
template <typename Vector>
class EulerBlock: public EulerBase<Eigen::VectorBlock<Vector, 3>>
{
public:
    typedef Eigen::VectorBlock<Vector, 3> Block;

    // constructor and destructor **********************************************
    EulerBlock(const Block& block): Block(block) { }
    virtual ~EulerBlock() {}
};



}

#endif
