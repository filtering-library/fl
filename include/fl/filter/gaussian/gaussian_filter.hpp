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
 * \file gaussian_filter.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__FILTER__GAUSSIAN__GAUSSIAN_FILTER_HPP
#define FL__FILTER__GAUSSIAN__GAUSSIAN_FILTER_HPP

namespace fl
{


struct FadingMemoryPolicy // None, alpha, decay
{
};

struct ValidationGatePolicy // Eucledian, ellipsoid
{
};

struct UpdatePolicy // normal, joseph
{

};


}

#include <fl/filter/gaussian/gaussian_filter_kf.hpp>
#include <fl/filter/gaussian/gaussian_filter_ukf.hpp>
#include <fl/filter/gaussian/gaussian_filter_ukf_npn_aon.hpp>

#endif
