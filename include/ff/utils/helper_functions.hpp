/*************************************************************************
This software allows for filtering in high-dimensional observation and
state spaces, as described in

M. Wuthrich, P. Pastor, M. Kalakrishnan, J. Bohg, and S. Schaal.
Probabilistic Object Tracking using a Range Camera
IEEE/RSJ Intl Conf on Intelligent Robots and Systems, 2013

In a publication based on this software pleace cite the above reference.


Copyright (C) 2014  Manuel Wuthrich

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*************************************************************************/

#ifndef FAST_FILTERING_UTILS_HELPER_FUNCTIONS_HPP
#define FAST_FILTERING_UTILS_HELPER_FUNCTIONS_HPP

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
