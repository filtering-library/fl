/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Max-Planck-Institute for Intelligent Systems,
 *                     University of Southern California
 *    Jan Issac (jan.issac@gmail.com)
 *    Manuel Wuthrich (manuel.wuthrich@gmail.com)
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
 * @author Jan Issac (jan.issac@gmail.com)
 * @author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * Max-Planck-Institute for Intelligent Systems, University of Southern California
 */

#ifndef FAST_FILTERING_MODELS_PROCESS_MODELS_LIBEAR_GAUSSIAN_PROCESS_MODEL_HPP
#define FAST_FILTERING_MODELS_PROCESS_MODELS_LIBEAR_GAUSSIAN_PROCESS_MODEL_HPP

#include <fast_filtering/utils/traits.hpp>
#include <fast_filtering/distributions/gaussian.hpp>
#include <fast_filtering/models/process_models/interfaces/stationary_process_model.hpp>

namespace ff
{

// Forward declarations
template <typename State, typename Input_> class LinearGaussianProcessModel;

template <typename State_, typename Input_>
struct Traits<LinearGaussianProcessModel<State_, Input_>>
{
    typedef Gaussian<State_> GaussianBase;
    typedef StationaryProcessModel<State_, Input_> ProcessModelBase;

    typedef State_ State;
    typedef Input_ Input;
    typedef typename Traits<GaussianBase>::Scalar Scalar;
    typedef typename Traits<GaussianBase>::Operator Operator;
    typedef typename Traits<GaussianBase>::Noise Noise;

    typedef Eigen::Matrix<Scalar,
                          State::SizeAtCompileTime,
                          State::SizeAtCompileTime> DynamicsMatrix;
};


template <typename State_, typename Input_ = internal::Empty>
class LinearGaussianProcessModel:
    public Traits<LinearGaussianProcessModel<State_, Input_>>::ProcessModelBase,
    public Traits<LinearGaussianProcessModel<State_, Input_>>::GaussianBase
{
public:
    typedef LinearGaussianProcessModel<State_, Input_> This;

    typedef typename Traits<This>::State State;
    typedef typename Traits<This>::Input Input;
    typedef typename Traits<This>::Noise Noise;
    typedef typename Traits<This>::Scalar Scalar;
    typedef typename Traits<This>::Operator Operator;
    typedef typename Traits<This>::DynamicsMatrix DynamicsMatrix;

    using Traits<This>::GaussianBase::Mean;
    using Traits<This>::GaussianBase::Covariance;
    using Traits<This>::GaussianBase::Dimension;

public:
    LinearGaussianProcessModel(
            const Operator& noise_covariance,
            const size_t dimension = State::SizeAtCompileTime):
        Traits<This>::GaussianBase(dimension),
        A_(DynamicsMatrix::Identity(Dimension(), Dimension())),
        delta_time_(1.)
    {
        Covariance(noise_covariance);
    }

    ~LinearGaussianProcessModel() { }

    virtual State MapStandardGaussian(const Noise& sample) const
    {
        return Mean() + delta_time_ * this->SquareRoot() * sample;
    }

    virtual void Condition(const double& delta_time,
                           const State& x,
                           const Input& u = Input())
    {
        delta_time_ = delta_time;                

        Mean(A_ * x);
    }

    virtual const DynamicsMatrix& A() const
    {
        return A_;
    }

    virtual void A(const DynamicsMatrix& dynamics_matrix)
    {
        A_ = dynamics_matrix;
    }

    virtual size_t InputDimension() const
    {
        return 0;
    }

protected:
    DynamicsMatrix A_;
    double delta_time_;    
};

}

#endif
