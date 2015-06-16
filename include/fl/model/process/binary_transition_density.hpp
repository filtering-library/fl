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
 * \file linear_process_model.hpp
 * \date June 2015
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#ifndef FL__MODEL__PROCESS__BINARY_TRANSITION_DENSITY_HPP
#define FL__MODEL__PROCESS__BINARY_TRANSITION_DENSITY_HPP

#include <fl/util/types.hpp>
#include <iostream>



namespace fl
{

class BinaryTransitionDensity /// \todo not really a density since it is discrete
{
public:
    BinaryTransitionDensity(const FloatingPoint& p_1to1,
                            const FloatingPoint& p_0to1):   p_1to1_(p_1to1),
                                                            p_0to1_(p_0to1)
    {
        /// \todo: this case could be handled properly, but it would be
        /// a little bit more involved
        if(p_0to1 > p_1to1)
        {
            std::cout << "the case of p_0to1 > p_1to1 is not handled in the "
                      << "binary transition density." << std::endl;
        }
    }

    FloatingPoint probability(const bool& next_state,
                              const bool& state,
                              const FloatingPoint& dt)
    {
        FloatingPoint a = std::pow(p_1to1_-p_0to1_, dt);
        FloatingPoint prob_1 =  a * FloatingPoint(state)
                                + (a - 1.) * p_0to1_ / (p_1to1_ - 1. - p_0to1_);

        // limit of p_0to1_ -> 0 and p_1to1_ -> 1
        if(!std::isfinite(prob_1))
            prob_1 = FloatingPoint(state);

        if(next_state == 1)
            return prob_1;
        else
            return 1. - prob_1;
    }


private:
    FloatingPoint p_1to1_;
    FloatingPoint p_0to1_;
};


}

#endif
