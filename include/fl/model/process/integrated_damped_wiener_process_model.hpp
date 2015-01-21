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

#ifndef FAST_FILTERING_MODELS_PROCESS_MODELS_INTEGRATED_DAMPED_WIENER_PROCESS_MODEL_HPP
#define FAST_FILTERING_MODELS_PROCESS_MODELS_INTEGRATED_DAMPED_WIENER_PROCESS_MODEL_HPP

#include <fl/util/math.hpp>
#include <fl/util/assertions.hpp>
#include <fl/distribution/gaussian.hpp>
#include <fl/distribution/interface/standard_gaussian_mapping.hpp>
#include <fl/model/process/process_model_interface.hpp>
#include <ff/models/process_models/damped_wiener_process_model.hpp>


//TODO: THIS IS A LINEAR GAUSSIAN PROCESS, THIS CLASS SHOULD DISAPPEAR


namespace fl
{

/// \todo MISSING DOC. MISSING UTESTS

// Forward declarations
template <typename State> class IntegratedDampedWienerProcessModel;

/**
 * IntegratedDampedWienerProcess distribution traits specialization
 * \internal
 */
template <typename State_>
struct Traits<IntegratedDampedWienerProcessModel<State_>>
{
    enum
    {
        STATE_DIMENSION = State_::SizeAtCompileTime,
        DEGREE_OF_FREEDOM = STATE_DIMENSION != -1 ? STATE_DIMENSION/2 : -1
    };

    typedef State_                                      State;
    typedef typename State::Scalar                      Scalar;
    typedef Eigen::Matrix<Scalar, DEGREE_OF_FREEDOM, 1> Input;
    typedef Eigen::Matrix<Scalar, DEGREE_OF_FREEDOM, 1> Noise;

    typedef ProcessModelInterface<State, Noise, Input> ProcessInterfaceBase;
    typedef StandardGaussianMapping<State, Noise>      GaussianMappingBase;

    typedef Eigen::Matrix<Scalar, DEGREE_OF_FREEDOM, 1> WienerProcessState;
    typedef DampedWienerProcessModel<WienerProcessState>     DampedWienerProcessType;
    typedef Gaussian<Noise>                             GaussianType;

    typedef typename GaussianType::SecondMoment      SecondMoment;
};

/**
 * \class IntegratedDampedWienerProcess
 *
 * \ingroup process_models
 */
template <typename State_>
class IntegratedDampedWienerProcessModel:
        public Traits<IntegratedDampedWienerProcessModel<State_>>::ProcessInterfaceBase,
        public Traits<IntegratedDampedWienerProcessModel<State_>>::GaussianMappingBase
{
public:
    typedef IntegratedDampedWienerProcessModel<State_> This;

    typedef typename Traits<This>::Scalar     Scalar;
    typedef typename Traits<This>::State      State;
    typedef typename Traits<This>::Input      Input;
    typedef typename Traits<This>::SecondMoment   SecondMoment;
    typedef typename Traits<This>::Noise      Noise;

    typedef typename Traits<This>::GaussianType GaussianType;
    typedef typename Traits<This>::DampedWienerProcessType DampedWienerProcessType;

    enum
    {
        STATE_DIMENSION = Traits<This>::STATE_DIMENSION,
        DEGREE_OF_FREEDOM = Traits<This>::DEGREE_OF_FREEDOM
    };

public:
    IntegratedDampedWienerProcessModel(
            const unsigned& degree_of_freedom = DEGREE_OF_FREEDOM):
        Traits<This>::GaussianMappingBase(degree_of_freedom),
        velocity_distribution_(degree_of_freedom),
        position_distribution_(degree_of_freedom)
    {
        static_assert_base(State, Eigen::Matrix<Scalar, STATE_DIMENSION, 1>);

        static_assert(
                STATE_DIMENSION % 2 == 0 || STATE_DIMENSION == Eigen::Dynamic,
                "Dimension must be a multitude of 2");
    }

    virtual ~IntegratedDampedWienerProcessModel() { }

    virtual State map_standard_normal(const Noise& sample) const
    {
        State state(StateDimension());
        state.topRows(InputDimension())     = position_distribution_.map_standard_normal(sample);
        state.bottomRows(InputDimension())  = velocity_distribution_.map_standard_normal(sample);
        return state;
    }


    virtual void condition(const Scalar&  delta_time,
                           const State&  state,
                           const Input&   input)
    {
        position_distribution_.mean(mean(state.topRows(InputDimension()), // position
                                                state.bottomRows(InputDimension()), // velocity
                                                input, // acceleration
                                                delta_time));
        position_distribution_.covariance(covariance(delta_time));

        velocity_distribution_.condition(delta_time, state.bottomRows(InputDimension()), input);
    }

    virtual State predict_state(double delta_time,
                                const State& state,
                                const Noise& noise,
                                const Input& input)
    {

    }

    virtual size_t state_dimension() const
    {
        return this->standard_variate_dimension() * 2;
    }

    virtual size_t noise_dimension() const
    {
        return this->standard_variate_dimension();
    }

    virtual size_t input_dimension() const
    {
        return this->standard_variate_dimension();
    }

    virtual void Parameters(
            const double& damping,
            const SecondMoment& acceleration_covariance)
    {
        damping_ = damping;
        acceleration_covariance_ = acceleration_covariance;

        velocity_distribution_.Parameters(damping, acceleration_covariance);
    }


    virtual unsigned InputDimension() const
    {
        return this->standard_variate_dimension();
    }

    virtual unsigned StateDimension() const
    {
        return this->standard_variate_dimension() * 2;
    }

private:
    Input mean(const Input& state,
                   const Input& velocity,
                   const Input& acceleration,
                   const double& delta_time)
    {
        Input mean;
        mean = state +
                (exp(-damping_ * delta_time) + damping_*delta_time  - 1.0)/pow(damping_, 2)
                * acceleration + (1.0 - exp(-damping_*delta_time))/damping_  * velocity;

        if(!std::isfinite(mean.norm()))
            mean = state +
                    0.5*delta_time*delta_time*acceleration +
                    delta_time*velocity;

        return mean;
    }

    SecondMoment covariance(const Scalar& delta_time)
    {
        // the first argument to the gamma function should be equal to zero, which would not cause
        // the gamma function to diverge as long as the second argument is not zero, which will not
        // be the case. boost however does not accept zero therefore we set it to a very small
        // value, which does not make a bit difference for any realistic delta_time
        Scalar factor =
               (-1.0 + exp(-2.0*damping_*delta_time))/(8.0*pow(damping_, 3)) +
                (2.0 - exp(-2.0*damping_*delta_time))/(4.0*pow(damping_, 2)) * delta_time +
                (-1.5 + fl::GAMMA + fl::igamma(0., 2.0*damping_*delta_time) +
                 log(2.0*damping_*delta_time))/(2.0*damping_)*pow(delta_time, 2);
        if(!std::isfinite(factor))
            factor = 1.0/3.0 * pow(delta_time, 3);

        return factor * acceleration_covariance_;
    }

private:
    DampedWienerProcessType velocity_distribution_;

    // conditionals
    GaussianType position_distribution_;

    // parameters
    Scalar damping_;
    SecondMoment acceleration_covariance_;
};

}

#endif
