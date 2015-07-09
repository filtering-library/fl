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

class EulerVector: public Eigen::Matrix<Real, 3, 1>
{
public:
    typedef Eigen::Matrix<Real, 3, 1>   Vector;

    // rotation types
    typedef Eigen::AngleAxis<Real>    AngleAxis;
    typedef Eigen::Quaternion<Real>   Quaternion;
    typedef Eigen::Matrix<Real, 3, 3> RotationMatrix;

    // constructor and destructor
    EulerVector() { }
    template <typename T> EulerVector(const Eigen::MatrixBase<T>& vector)
    {
        *this = vector;
    }

    virtual ~EulerVector() {}

    /// operators *************************************************************
    template <typename T>
    void operator = (const Eigen::MatrixBase<T>& vector)
    {
        *((Vector*)(this)) = vector;
    }
    EulerVector operator * (const EulerVector& factor)
    {
        EulerVector product;
        product.quaternion(this->quaternion() * factor.quaternion());
        return product;
    }

    /// accessor **************************************************************
    virtual Real angle() const
    {
        return this->norm();
    }
    virtual Vector axis() const
    {
        Vector  ax =  this->normalized();

        if(!std::isfinite(ax.sum()))
        {
            return Vector(1,0,0);
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

    virtual EulerVector inverse() const
    {
        return - (*this);
    }

    /// mutators **************************************************************
    virtual void angle_axis(const Real& angle, const Vector& axis)
    {
        *((Vector*)(this)) = angle * axis;
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

}

#endif
