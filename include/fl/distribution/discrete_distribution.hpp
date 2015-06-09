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
 * \file gaussian.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#ifndef FL__DISTRIBUTION__DISCRETE_DISTRIBUTION_HPP
#define FL__DISTRIBUTION__DISCRETE_DISTRIBUTION_HPP



#include <Eigen/Dense>

#include <vector>
#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <cmath>

#include <ctime>
#include <fstream>

#include <fl/util/random.hpp>
#include <fl/util/math.hpp>

namespace fl
{

namespace hf
{

// sampling class >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class DiscreteDistribution
{
public:
    template <typename T> DiscreteDistribution(std::vector<T> log_prob)
        : generator_(RANDOM_SEED),
          uniform_distribution_(0., 1.)
    {
        // substract max to avoid numerical issues

        // replaced by std::max_element
        //double max = hf::bound_value(log_prob, true);
        double max = *(std::max_element(log_prob.begin(), log_prob.end()));

        for(int i = 0; i < int(log_prob.size()); i++)
            log_prob[i] -= max;

        // compute probabilities
        std::vector<double> prob(log_prob.size());
        double sum = 0;
        for(int i = 0; i < int(log_prob.size()); i++)
        {
            prob[i] = std::exp(log_prob[i]);
            sum += prob[i];
        }
        for(int i = 0; i < int(prob.size()); i++)
            prob[i] /= sum;

        // compute the cumulative probability
        cumulative_prob_.resize(prob.size());
        cumulative_prob_[0] = prob[0];
        for(size_t i = 1; i < prob.size(); i++)
            cumulative_prob_[i] = cumulative_prob_[i-1] + prob[i];
    }

    ~DiscreteDistribution() {}

    int sample()
    {
        return map_standard_uniform(uniform_distribution_(generator_));
    }

    int map_standard_normal(double gaussian_sample) const
    {
        double uniform_sample =
                0.5 * (1.0 + std::erf(gaussian_sample / std::sqrt(2.0)));

        return map_standard_uniform(uniform_sample);
    }

    int map_standard_uniform(double uniform_sample) const
    {
        std::vector<double>::const_iterator iterator =
                                 std::lower_bound(cumulative_prob_.begin(),
                                                  cumulative_prob_.end(),
                                                  uniform_sample);

        return iterator - cumulative_prob_.begin();
    }

private:
    std::vector<double> cumulative_prob_;

    fl::mt11213b generator_;
    std::uniform_real_distribution<double> uniform_distribution_;
};

}

}

#endif
