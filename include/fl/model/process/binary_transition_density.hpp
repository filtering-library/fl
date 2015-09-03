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
 * \file linear_state_transition_model.hpp
 * \date June 2015
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#pragma once


#include <fl/util/types.hpp>
#include <iostream>



namespace fl
{

class BinaryTransitionDensity /// \todo not really a density since it is discrete
{
public:
    BinaryTransitionDensity(const Real& p_1to1,
                            const Real& p_0to1):   p_1to1_(p_1to1),
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

    Real probability(const bool& next_state,
                              const bool& state,
                              const Real& dt)
    {
        Real a = std::pow(p_1to1_ - p_0to1_, dt);
        Real prob_1 =  a * Real(state)
                                + (a - 1.) * p_0to1_ / (p_1to1_ - 1. - p_0to1_);

        // limit of p_0to1_ -> 0 and p_1to1_ -> 1
        if(!std::isfinite(prob_1)) prob_1 = Real(state);

        return next_state ? prob_1 : 1. - prob_1;
    }


private:
    Real p_1to1_;
    Real p_0to1_;
};


}


