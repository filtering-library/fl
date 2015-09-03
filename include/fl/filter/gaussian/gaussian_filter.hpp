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

#pragma once


#include "transform/unscented_transform.hpp"
#include "transform/monte_carlo_transform.hpp"

#include "quadrature/sigma_point_quadrature.hpp"
#include "quadrature/unscented_quadrature.hpp"

#include "gaussian_filter_linear.hpp"
#include "gaussian_filter_nonlinear.hpp"
#include "gaussian_filter_nonlinear_generic.hpp"
