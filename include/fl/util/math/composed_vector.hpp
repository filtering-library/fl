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

#pragma once



#include <Eigen/Dense>
#include <fl/util/types.hpp>

namespace fl
{

template <typename Block, typename Vector>
class ComposedVector: public Vector
{
public:
    // constructor and destructor **********************************************
    ComposedVector() { }

    template <typename T>
    ComposedVector(const Eigen::MatrixBase<T>& vector): Vector(vector) { }

    virtual ~ComposedVector() noexcept {}

    // operators ***************************************************************
    template <typename T>
    void operator = (const Eigen::MatrixBase<T>& vector)
    {
        *((Vector*)(this)) = vector;
    }

    // accessors ***************************************************************
    const Block component(int index) const
    {
        return Block(*this, index * Block::SizeAtCompileTime);
    }
    int count() const
    {
        return this->size() / Block::SizeAtCompileTime;
    }

    // mutators ****************************************************************
    Block component(int index)
    {
        return Block(*this, index * Block::SizeAtCompileTime);
    }
    void recount(int new_count)
    {
        return this->resize(new_count * Block::SizeAtCompileTime);
    }
};

}


