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
 * \file multiple_of.hpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__UTIL__META__OPERATOR_MULTIPLE_OF_HPP
#define FL__UTIL__META__OPERATOR_MULTIPLE_OF_HPP

namespace fl
{

/**
 * \ingroup meta
 *
 * Same as CreateTypeSequence, however with a reversed parameter order. This is
 * an attempt to make the use of \c CreateTypeSequence more natural. It also
 * allows dynamic sizes if needed.
 */
template <typename Type, int Count = Eigen::Dynamic>
struct MultipleOf
    : CreateTypeSequence<Count, Type>
{
    template <typename InstanceType>
    explicit
    MultipleOf(InstanceType&& instance, int instance_count = Count)
        : instance(std::forward<InstanceType>(instance)),
          count(instance_count)
    { }

    Type instance;
    int count;
};

/**
 * \internal
 * \ingroup meta
 * \todo fix
 *
 * Creates an empty TypeSequence for dynamic count
 */
//template <typename Type>
//struct MultipleOf<Type, Eigen::Dynamic>
//    : CreateTypeSequence<Eigen::Dynamic, Type>
//{ };

}

#endif
