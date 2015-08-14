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
 * \date August 2015
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#ifndef FL__UTIL__MATH__MULTI_BLOCK_VECTOR_HPP
#define FL__UTIL__MATH__MULTI_BLOCK_VECTOR_HPP


#include <Eigen/Dense>
#include <fl/util/types.hpp>

namespace fl
{

template <typename Block, int N>
struct MultiBlockVectorTypes
{
    typedef Eigen::Matrix<Real, N == Eigen::Dynamic ?
                    Eigen::Dynamic : N * Block::SizeAtCompileTime(), 1>  Vector;

};


/// basic functionality for both vectors and blocks ****************************
// the block is assumed to be fixed size
template <typename Block, int N>
class MultiBlockVector: public MultiBlockVectorTypes<Block, N>::Vector
{
public:
    typedef typename MultiBlockVectorTypes<Block, N>::Vector Vector;

    // constructor and destructor **********************************************
    MultiBlockVector() { }

    MultiBlockVector(int n): Vector(n * Block::SizeAtCompileTime()) { }

    template <typename T>
    MultiBlockVector(const Eigen::MatrixBase<T>& vector): Vector(vector) { }

    virtual ~MultiBlockVector() {}

    // operators ***************************************************************
    template <typename T>
    void operator = (const Eigen::MatrixBase<T>& vector)
    {
        *((Vector*)(this)) = vector;
    }
};

}

#endif
