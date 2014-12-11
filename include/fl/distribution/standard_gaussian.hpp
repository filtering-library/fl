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
 * @date 2014
 * @author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * @author Jan Issac (jan.issac@gmail.com)
 * Max-Planck-Institute for Intelligent Systems, University of Southern California
 */

#ifndef FL__DISTRIBUTIONS__STANDARD_GAUSSIAN_HPP
#define FL__DISTRIBUTIONS__STANDARD_GAUSSIAN_HPP

#include <Eigen/Dense>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>

#include <fl/util/random_seed.hpp>
#include <fl/util/traits.hpp>
#include <fl/distribution/interface/sampling.hpp>
#include <fl/exception/exception.hpp>

namespace fl
{

template <typename Vector>
class StandardGaussian:
        public Sampling<Vector>
{
public:
    explicit StandardGaussian(const int dimension = DimensionOf<Vector>()):
        dimension_ (dimension),
        generator_(RANDOM_SEED),
        gaussian_distribution_(0.0, 1.0),
        gaussian_generator_(generator_, gaussian_distribution_)
    { }

    virtual ~StandardGaussian() { }

    virtual Vector Sample()
    {
        Vector gaussian_sample(Dimension());
        for (int i = 0; i < Dimension(); i++)
        {
            gaussian_sample(i) = gaussian_generator_();
        }

        return gaussian_sample;
    }

    virtual int Dimension() const
    {
        return dimension_;
    }

    virtual void Dimension(size_t new_dimension)
    {
        if (dimension_ == new_dimension) return;

        if (fl::IsFixed<Vector::SizeAtCompileTime>())
        {
            BOOST_THROW_EXCEPTION(
                fl::ResizingFixedSizeEntityException(dimension_,
                                                     new_dimension,
                                                     "Gaussian"));
        }

        dimension_ = new_dimension;
    }

private:
    int dimension_;
    boost::mt19937 generator_;
    boost::normal_distribution<> gaussian_distribution_;
    boost::variate_generator<
        boost::mt19937, boost::normal_distribution<>> gaussian_generator_;
};

// specialization for scalar
template<>
class StandardGaussian<double>: public Sampling<double>
{
public:
    StandardGaussian():
        generator_(RANDOM_SEED),
        gaussian_distribution_(0.0, 1.0),
        gaussian_generator_(generator_, gaussian_distribution_) { }

    virtual ~StandardGaussian() { }

    virtual double Sample()
    {
        return gaussian_generator_();
    }

    virtual int Dimension() const
    {
        return 1;
    }

private:
    boost::mt19937 generator_;
    boost::normal_distribution<> gaussian_distribution_;
    boost::variate_generator<
        boost::mt19937, boost::normal_distribution<>> gaussian_generator_;
};

}

#endif
