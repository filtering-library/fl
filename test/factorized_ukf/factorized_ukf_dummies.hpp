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
 * Max-Planck-Institute for Intelligent Systems, University of Southern California
 */

#ifndef FAST_FILTERING_TESTS_DUMMY_MODELS_HPP
#define FAST_FILTERING_TESTS_DUMMY_MODELS_HPP

#include <Eigen/Dense>

#include <fl/util/traits.hpp>
#include <fl/distribution/interface/standard_gaussian_mapping.hpp>
#include <fl/model/process/process_model_interface.hpp>
#include <fl/model/observation/observation_model_interface.hpp>
#include <ff/filters/deterministic/factorized_unscented_kalman_filter.hpp>

template <typename State> class ProcessModelDummy;

namespace fl
{

template <typename State_> struct Traits<ProcessModelDummy<State_>>
{
    typedef State_ State;
    typedef State_ Noise;
    typedef State_ Input;
    typedef typename State::Scalar Scalar;    

    typedef ProcessModelInterface<State, Noise, Input> ProcessModelBase;
    typedef StandardGaussianMapping<State_, State_> GaussianMappingBase;
};
}

template <typename State>
class ProcessModelDummy:
        public fl::Traits<ProcessModelDummy<State>>::ProcessModelBase,
        public fl::Traits<ProcessModelDummy<State>>::GaussianMappingBase
{
public:
    typedef ProcessModelDummy<State> This;

    typedef typename fl::Traits<This>::Scalar Scalar;
    typedef typename fl::Traits<This>::Noise  Noise;
    typedef typename fl::Traits<This>::Input  Input;

    virtual State predict_state(double delta_time,
                                const State& state,
                                const Noise& noise,
                                const Input& input = Input())
    {

    }

    virtual void condition(const double& delta_time,
                           const State& state,
                           const Input& input)
    {

    }

    virtual State map_standard_normal(const Noise& sample) const
    {
        return State();
    }

    virtual size_t state_dimension() const
    {
        return State::SizeAtCompileTime;
    }

    virtual size_t input_dimension() const
    {
        return state_dimension();
    }

    virtual size_t noise_dimension() const
    {
        return state_dimension();
    }
};



template <typename State, typename Observation> class ObservationModelDummy;

namespace fl
{

template <typename State_, typename Observation_>
struct Traits<ObservationModelDummy<State_, Observation_>>
{
    typedef State_ State;
    typedef State_ Noise;
    typedef Observation_ Observation;
    typedef typename State::Scalar Scalar;

    typedef ObservationModelInterface<State, Observation, Noise> ObservationModelBase;
};
}

template <typename State, typename Obsrv>
class ObservationModelDummy
        : public fl::Traits<
                     ObservationModelDummy<State, Obsrv>
                 >::ObservationModelBase

{
public:
    typedef ObservationModelDummy<State, Obsrv> This;

    typedef typename fl::Traits<This>::Scalar Scalar;
    typedef typename fl::Traits<This>::Noise  Noise;
    typedef typename fl::Traits<This>::Observation Observation;

    virtual Observation predict_observation(const State& state,
                                            const Noise& noise,
                                            double delta_time)
    {

    }

    virtual size_t observation_dimension() const
    {
        return Observation::RowsAtCompileTime;
    }

    virtual size_t state_dimension() const
    {
        return observation_dimension();
    }

    virtual size_t noise_dimension() const
    {
        return observation_dimension();
    }
};

#endif

