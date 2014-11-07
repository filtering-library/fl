/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Max-Planck-Institute for Intelligent Systems,
 *                     University of Southern California
 *    Manuel Wuthrich (manuel.wuthrich@gmail.com)
 *    Jan Issac (jan.issac@gmail.com)
 *
 *
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

/**
 * @date 05/25/2014
 * @author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * @author Jan Issac (jan.issac@gmail.com)
 * Max-Planck-Institute for Intelligent Systems,
 *  University of Southern California
 */


#ifndef FAST_FILTERING_DISTRIBUTIONS_INTERFACES_EVAUATION_HPP
#define FAST_FILTERING_DISTRIBUTIONS_INTERFACES_EVAUATION_HPP

#include <cmath>
#include <fast_filtering/utils/traits.hpp>
#include <fast_filtering/distributions/interfaces/unnormalized_evaluation.hpp>

namespace ff
{

/**
 * \interface Evaluation
 * \brief Distribution evaulation interface
 *
 * \tparam Vector   Random variable type
 * \tparam Scalar   Probability & log scalar type
 *
 * Evaluation provides the interface to determine the probability of the
 * underlying distribution at a given sample. Evaluation is a subset of
 * unnormalized distributions.
 */
template <typename Vector, typename Scalar>
class Evaluation:
        public UnnormalizedEvaulation<Vector, Scalar>
{
public:
    /**
     * \return Normalized probability of a given sample
     *
     * \param vector Sample to evaluate
     */
    virtual Scalar Probability(const Vector& vector) const
    {
        return std::exp(LogProbability(vector));
    }

    /**
     * \return Log of normalized probability of a given sample
     */
    virtual Scalar LogProbability(const Vector& vector) const = 0;

    /**
     * \{UnnormalizedEvaulation
     *   ::LogUnnormalizedProbability(const Vector&) const\}
     */
    virtual Scalar LogUnnormalizedProbability(const Vector& vector) const
    {
        return LogProbability(vector);
    }

    /**
     * \brief Overridable default destructor
     */
    virtual ~Evaluation() { }
};

}

#endif
