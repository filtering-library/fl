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

#include <vector>
#include <algorithm>
#include <iostream>
#include <limits>
#include <Eigen/Dense>
#include <cmath>

#include <ctime>
#include <fstream>



#include <boost/lexical_cast.hpp>
#include <boost/random/lagged_fibonacci.hpp>

#include <fl/util/random_seed.hpp>



// TODO: THIS HAS TO BE CLEANED, POSSIBLY SPLIT INTO SEVERAL FILES

namespace fl
{

namespace hf
{


// use std::min_element & std::max_element instead. better API if split into two functions
template <typename T> int BoundIndex(const std::vector<T> &values, bool bound_type) // bound type 1 for max and 0 for min
{
	int BoundIndex = 0;
	T bound_value = bound_type ? -std::numeric_limits<T>::max() : std::numeric_limits<T>::max();

	for(int i = 0; i < int(values.size()); i++)
		if(bound_type ? (values[i] > bound_value) : (values[i] < bound_value) )
		{
			BoundIndex = i;
			bound_value = values[i];
		}

	return BoundIndex;
}

// use std::min_element & std::max_element instead
template <typename T> T bound_value(const std::vector<T> &values, bool bound_type) // bound type 1 for max and 0 for min
{
	return values[BoundIndex(values, bound_type)];
}

// use std::transform or for_each instead
template <typename Tin, typename Tout>
std::vector<Tout> Apply(const std::vector<Tin> &input, Tout(*f)(Tin))
{
	std::vector<Tout> output(input.size());
	for(size_t i = 0; i < output.size(); i++)
		output[i] = (*f)(input[i]);

	return output;
}

// sampling class >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class DiscreteDistribution
{
public:
    template <typename T> DiscreteDistribution(std::vector<T> log_prob)
    {
        uniform_sampler_.seed(RANDOM_SEED);

        // substract max to avoid numerical issues
        double max = hf::bound_value(log_prob, true);
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

    int Sample()
    {
        return MapStandardUniform(uniform_sampler_());
    }

    int MapStandardGaussian(double gaussian_sample) const
    {
        double uniform_sample =
                0.5 * (1.0 + std::erf(gaussian_sample / std::sqrt(2.0)));
        return MapStandardUniform(uniform_sample);
    }

    int MapStandardUniform(double uniform_sample) const
    {
        std::vector<double>::const_iterator iterator =
                                 std::lower_bound(cumulative_prob_.begin(),
                                                  cumulative_prob_.end(),
                                                  uniform_sample);

        return iterator - cumulative_prob_.begin();
    }

private:
    boost::lagged_fibonacci607  uniform_sampler_;
    std::vector<double>         cumulative_prob_;
};

}

}

#endif
