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
 * \file random_seed.hpp
 * \date 2014
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#ifndef FL__UTIL__RANDOM_SEED_HPP
#define FL__UTIL__RANDOM_SEED_HPP

#include <ctime>

#define RANDOM_SEED (unsigned) std::time(0) // 1

#endif
