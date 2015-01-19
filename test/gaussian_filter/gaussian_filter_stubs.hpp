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
 * Max-Planck-Institute for Intelligent Systems,
 * University of Southern California
 */

#include <gtest/gtest.h>

#include <fl/util/traits.hpp>
#include <fl/model/process/process_model_interface.hpp>
#include <fl/model/observation/observation_model_interface.hpp>

template <typename State, typename Noise, typename Input>
class ProcessModelStub;

namespace fl
{
template <typename State_, typename Noise_, typename Input_>
struct Traits<ProcessModelStub<State_, Noise_, Input_>>
{
    typedef State_ State;
    typedef Noise_ Noise;
    typedef Input_ Input;
};
}

template <typename State, typename Noise, typename Input>
class ProcessModelStub
        : public fl::ProcessModelInterface<State, Noise, Input>
{
public:
    ProcessModelStub(size_t state_dimension = fl::DimensionOf<State>(),
                     size_t noise_dimension = fl::DimensionOf<Noise>(),
                     size_t input_dimension = fl::DimensionOf<Input>())
        : state_dimension_(state_dimension),
          noise_dimension_(noise_dimension),
          input_dimension_(input_dimension)
    {

    }

    virtual void condition(const double& delta_time,
                           const State& state,
                           const Input& input)
    {

    }

    virtual State predict_state(double delta_time,
                                const State& state,
                                const Noise& noise,
                                const Input& input)
    {

    }

    size_t state_dimension() const { return state_dimension_; }
    size_t noise_dimension() const { return noise_dimension_; }
    size_t input_dimension() const { return input_dimension_; }

protected:
    size_t state_dimension_;
    size_t noise_dimension_;
    size_t input_dimension_;
};

template <typename State, typename Obsrv, typename Noise>
class ObservationModelStub;

namespace fl
{
template <typename State_, typename Obsrv_, typename Noise_>
struct Traits<ObservationModelStub<State_, Obsrv_, Noise_>>
{
    typedef State_ State;
    typedef Noise_ Noise;
    typedef Obsrv_ Observation;
};
}

template <typename State, typename Obsrv, typename Noise>
class ObservationModelStub
        : public fl::ObservationModelInterface<State, Obsrv, Noise>
{
public:
    ObservationModelStub(size_t state_dimension = fl::DimensionOf<State>(),
                         size_t obsrv_dimension = fl::DimensionOf<Obsrv>(),
                         size_t noise_dimension = fl::DimensionOf<Noise>())
        : state_dimension_(state_dimension),
          obsrv_dimension_(obsrv_dimension),
          noise_dimension_(noise_dimension)
    {

    }


    virtual Obsrv predict_observation(const State& state,
                                      const Noise& noise)
    {

    }

    size_t state_dimension() const { return state_dimension_; }
    size_t obsrv_dimension() const { return obsrv_dimension_; }
    size_t noise_dimension() const { return noise_dimension_; }

protected:
    size_t state_dimension_;
    size_t obsrv_dimension_;
    size_t noise_dimension_;
};
