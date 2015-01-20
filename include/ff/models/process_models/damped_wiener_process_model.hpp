/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Max-Planck-Institute for Intelligent Systems,
 *                     University of Southern California
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
 * @date 05/25/2014
 * @author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * Max-Planck-Institute for Intelligent Systems, University of Southern California
 */

#ifndef FAST_FILTERING_MODELS_PROCESS_MODELS_DAMPED_WIENER_PROCESS_MODEL_HPP
#define FAST_FILTERING_MODELS_PROCESS_MODELS_DAMPED_WIENER_PROCESS_MODEL_HPP

//TODO: THIS IS A LINEAR GAUSSIAN PROCESS, THIS CLASS SHOULD DISAPPEAR

#include <Eigen/Dense>

#include <fl/util/assertions.hpp>
#include <fl/distribution/gaussian.hpp>
#include <fl/model/process/process_model_interface.hpp>



namespace fl
{

/// \todo MISSING DOC. MISSING UTESTS

// Forward declarations
template <typename State> class DampedWienerProcessModel;

/**
 * DampedWienerProcess distribution traits specialization
 * \internal
 */
template <typename State_>
struct Traits<DampedWienerProcessModel<State_>>
{
    typedef State_                                              State;
    typedef typename State::Scalar                              Scalar;
    typedef Eigen::Matrix<Scalar, State::SizeAtCompileTime, 1>  Input;
    typedef Eigen::Matrix<Scalar, State::SizeAtCompileTime, 1>  Noise;

    typedef Gaussian<Noise> GaussianType;
    typedef typename GaussianType::Operator Operator;

    typedef StandardGaussianMapping<State, Noise> GaussianMappingBase;
    typedef ProcessModelInterface<State, Noise, Input> ProcessInterfaceBase;
};

/**
 * \class DampedWienerProcess
 *
 * \ingroup process_models
 */
template <typename State>
class DampedWienerProcessModel:
        public Traits<DampedWienerProcessModel<State>>::GaussianMappingBase,
        public Traits<DampedWienerProcessModel<State>>::ProcessInterfaceBase
{
public:
    typedef DampedWienerProcessModel<State> This;

    typedef typename Traits<This>::Scalar         Scalar;
    typedef typename Traits<This>::Operator       Operator;
    typedef typename Traits<This>::Input          Input;
    typedef typename Traits<This>::Noise          Noise;
    typedef typename Traits<This>::GaussianType   GaussianType;

public:
    explicit DampedWienerProcessModel(size_t dim = DimensionOf<State>())
        : Traits<This>::GaussianMappingBase(dim),
          gaussian_(dim)
    {
    }

    virtual ~DampedWienerProcessModel() { }

    virtual State map_standard_normal(const Noise& sample) const
    {
        return gaussian_.map_standard_normal(sample);
    }

    virtual void condition(const Scalar&  delta_time,
                           const State&  state,
                           const Input&   input)
    {
        gaussian_.mean(mean(delta_time, state, input));
        gaussian_.diagonal_covariance(covariance(delta_time));
    }

    virtual void Parameters(const Scalar& damping,
                            const Operator& noise_covariance)
    {
        damping_ = damping;
        noise_covariance_ = noise_covariance;
    }

    virtual unsigned dimension() const
    {
        return this->standard_variate_dimension();
    }

    /* interface API */
    virtual State predict_state(double delta_time,
                                const State& state,
                                const Noise& noise,
                                const Input& input)
    {
        condition(delta_time, state, input);

        return map_standard_normal(noise);
    }

    virtual size_t state_dimension() const
    {
        return this->standard_variate_dimension();
    }

    virtual size_t noise_dimension() const
    {
        return this->standard_variate_dimension();
    }

    virtual size_t input_dimension() const
    {
        return this->standard_variate_dimension();
    }

private:
    State mean(const Scalar& delta_time,
                    const State& state,
                    const Input& input)
    {
        if(damping_ == 0)
            return state + delta_time * input;

        State state_expectation = (1.0 - exp(-damping_*delta_time)) / damping_ * input +
                                              exp(-damping_*delta_time)  * state;

        // if the damping_ is too small, the result might be nan, we thus return the limit for damping_ -> 0
        if(!std::isfinite(state_expectation.norm()))
            state_expectation = state + delta_time * input;

        return state_expectation;
    }

    Operator covariance(const Scalar& delta_time)
    {
        if(damping_ == 0)
            return delta_time * noise_covariance_;

        Scalar factor = (1.0 - exp(-2.0*damping_*delta_time))/(2.0*damping_);
        if(!std::isfinite(factor))
            factor = delta_time;

        return factor * noise_covariance_;
    }

private:
    // conditional
    GaussianType gaussian_;

    // parameters
    Scalar damping_;
    Operator noise_covariance_;
};

}

#endif
