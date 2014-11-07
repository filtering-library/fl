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

#ifndef FAST_FILTERING_MODELS_PROCESS_MODELS_LINEAR_GAUSSIAN_OBSERVATION_MODEL_HPP
#define FAST_FILTERING_MODELS_PROCESS_MODELS_LINEAR_GAUSSIAN_OBSERVATION_MODEL_HPP

#include <fast_filtering/utils/traits.hpp>
#include <fast_filtering/distributions/gaussian.hpp>
#include <fast_filtering/models/process_models/interfaces/stationary_process_model.hpp>

namespace ff
{

// Forward declarations
template <typename Observation_, typename State_> class LinearGaussianObservationModel;

template <typename Observation_, typename State_>
struct Traits<LinearGaussianObservationModel<Observation_, State_>>
{
    typedef Observation_ Observation;
    typedef State_ State;
    typedef Gaussian<Observation_> GaussianBase;
    typedef typename Traits<GaussianBase>::Scalar Scalar;
    typedef typename Traits<GaussianBase>::Operator Operator;
    typedef typename Traits<GaussianBase>::Noise Noise;
    typedef Eigen::Matrix<Scalar,
                          Observation::SizeAtCompileTime,
                          State::SizeAtCompileTime> SensorMatrix;
};

template <typename Observation_,typename State_>
class LinearGaussianObservationModel:
    public Gaussian<Observation_>
{
public:
    typedef LinearGaussianObservationModel<Observation_, State_> This;

    typedef typename Traits<This>::State State;
    typedef typename Traits<This>::Observation Observation;
    typedef typename Traits<This>::Noise Noise;
    typedef typename Traits<This>::Scalar Scalar;
    typedef typename Traits<This>::Operator Operator;
    typedef typename Traits<This>::SensorMatrix SensorMatrix;

    using Traits<This>::GaussianBase::Mean;
    using Traits<This>::GaussianBase::Covariance;
    using Traits<This>::GaussianBase::Dimension;

public:
    LinearGaussianObservationModel(
            const Operator& noise_covariance,
            const size_t observation_dimension = Observation::SizeAtCompileTime,
            const size_t state_dimension = State::SizeAtCompileTime):
        Traits<This>::GaussianBase(observation_dimension),
        state_dimension_(state_dimension == Eigen::Dynamic? 0 : state_dimension),
        H_(SensorMatrix::Zero(Dimension(), StateDimension()))
    {
        Covariance(noise_covariance);
    }

    ~LinearGaussianObservationModel() { }

    virtual void Condition(const State& x)
    {
        Mean(H_ * x);
    }

    virtual const SensorMatrix& H() const
    {
        return H_;
    }

    virtual void H(const SensorMatrix& sensor_matrix)
    {
        H_ = sensor_matrix;
    }

    virtual size_t StateDimension() const
    {
        return state_dimension_;
    }

protected:
    size_t state_dimension_;
    SensorMatrix H_;
};


// Forward declarations
template <typename Observation_, typename State_a_, typename State_b_>
class FactorizedLinearGaussianObservationModel;

template <typename Observation_, typename State_a_, typename State_b_>
struct Traits<FactorizedLinearGaussianObservationModel<Observation_,
                                                      State_a_,
                                                      State_b_>>
{
    typedef Observation_ Observation;
    typedef State_a_ State_a;
    typedef State_b_ State_b;
    typedef Gaussian<Observation_> GaussianBase;
    typedef typename Traits<GaussianBase>::Scalar Scalar;
    typedef typename Traits<GaussianBase>::Operator Operator;
    typedef typename Traits<GaussianBase>::Noise Noise;
    typedef Eigen::Matrix<Scalar,
                          Observation::SizeAtCompileTime,
                          State_a::SizeAtCompileTime> SensorMatrix_a;
    typedef Eigen::Matrix<Scalar,
                          Observation::SizeAtCompileTime,
                          State_b::SizeAtCompileTime> SensorMatrix_b;
};

template <typename Observation_,typename State_a_, typename State_b_>
class FactorizedLinearGaussianObservationModel:
    public Traits<
               FactorizedLinearGaussianObservationModel<
                   Observation_, State_a_, State_b_>>::GaussianBase
{
public:
    typedef FactorizedLinearGaussianObservationModel<
                Observation_,
                State_a_,
                State_b_
            > This;

    typedef typename Traits<This>::Noise Noise;
    typedef typename Traits<This>::Scalar Scalar;
    typedef typename Traits<This>::Operator Operator;
    typedef typename Traits<This>::Observation Observation;
    typedef typename Traits<This>::State_a State_a;
    typedef typename Traits<This>::State_b State_b;
    typedef typename Traits<This>::SensorMatrix_a SensorMatrix_a;
    typedef typename Traits<This>::SensorMatrix_b SensorMatrix_b;

    using Traits<This>::GaussianBase::Mean;
    using Traits<This>::GaussianBase::Covariance;
    using Traits<This>::GaussianBase::Dimension;

public:
    FactorizedLinearGaussianObservationModel(
            const Operator& noise_covariance,
            const size_t observation_dimension = Observation::SizeAtCompileTime,
            const size_t state_a_dimension = State_a::SizeAtCompileTime,
            const size_t state_b_dimension = State_b::SizeAtCompileTime):
        Traits<This>::GaussianBase(observation_dimension),
        state_a_dimension_(state_a_dimension),
        state_b_dimension_(state_b_dimension),
        H_a_(SensorMatrix_a::Zero(Dimension(), State_a_Dimension())),
        H_b_(SensorMatrix_b::Zero(Dimension(), State_b_Dimension()))
    {
        Covariance(noise_covariance);
    }

    ~FactorizedLinearGaussianObservationModel() { }

    virtual Observation Predict(const Noise& noise) const
    {
        return MapStandardGaussian(noise);
    }

    virtual void Condition(const State_a& state_a,
                           const State_b& state_b,
                           size_t state_index,
                           size_t pixel_index)
    {
        Mean(H_a_ * state_a + H_b_ * state_b);
    }

    virtual const SensorMatrix_a& H_a() const
    {
        return H_a_;
    }

    virtual const SensorMatrix_b& H_b() const
    {
        return H_b_;
    }

    virtual void H_a(const SensorMatrix_a& sensor_matrix_a)
    {
        H_a_ = sensor_matrix_a;
    }

    virtual void H_b(const SensorMatrix_b& sensor_matrix_b)
    {
        H_b_ = sensor_matrix_b;
    }

    virtual size_t State_a_Dimension() const
    {
        return state_a_dimension_;
    }

    virtual size_t State_b_Dimension() const
    {
        return state_b_dimension_;
    }

protected:
    size_t state_a_dimension_;
    size_t state_b_dimension_;
    SensorMatrix_a H_a_;
    SensorMatrix_b H_b_;
};





//template <typename Observation_,typename State_a_, typename State_b_>
//class FactorizedLinearGaussianOservationModel2;


//template <typename Observation_, typename State_a_, typename State_b_>
//struct Traits<FactorizedLinearGaussianOservationModel2<Observation_,
//                                                       State_a_,
//                                                       State_b_>>
//{
//    typedef Observation_ Observation;

//    typedef Gaussian<Observation_> GaussianBase;
//    typedef typename Traits<GaussianBase>::Scalar Scalar;
//    typedef typename Traits<GaussianBase>::Operator Operator;
//    typedef typename Traits<GaussianBase>::Noise Noise;

//    typedef State_a_ State_a;
//    typedef State_b_ State_b;

//    enum
//    {
//        Dim_ab = (State_a::SizeAtCompileTime == Eigen::Dynamic ||
//                  State_b::SizeAtCompileTime == Eigen::Dynamic)
//                    ? Eigen::Dynamic
//                    : State_a::SizeAtCompileTime + State_b::SizeAtCompileTime
//    };

//    typedef Eigen::Matrix<Scalar, Dim_ab, 1> State_ab;
//    typedef Eigen::Matrix<Scalar, Dim_ab, Dim_ab> SensorMatrix_ab;

//    typedef Eigen::Matrix<Scalar,
//                          Observation::SizeAtCompileTime,
//                          State_a::SizeAtCompileTime> SensorMatrix_a;
//    typedef Eigen::Matrix<Scalar,
//                          Observation::SizeAtCompileTime,
//                          State_b::SizeAtCompileTime> SensorMatrix_b;

//    typedef LinearGaussianOservationModel<Observation, State_ab> Base;
//};



//template <typename Observation_,typename State_a_, typename State_b_>
//class FactorizedLinearGaussianOservationModel2:
//        public Traits<FactorizedLinearGaussianOservationModel2<
//                    Observation_, State_a_, State_b_>>::Base
//{
//public:
//    typedef Traits<FactorizedLinearGaussianOservationModel2<
//        Observation_, State_a_, State_b_>> Traits;

//    typedef typename Traits::State_a State_a;
//    typedef typename Traits::State_b State_b;
//    typedef typename Traits::Observation Observation;
//    typedef typename Traits::Noise Noise;
//    typedef typename Traits::Scalar Scalar;
//    typedef typename Traits::Operator Operator;
//    typedef typename Traits::SensorMatrix_a SensorMatrix_a;
//    typedef typename Traits::SensorMatrix_b SensorMatrix_b;

//    using Traits::GaussianBase::Mean;
//    using Traits::GaussianBase::Covariance;
//    using Traits::GaussianBase::Dimension;
//    using Traits::Base::MapStandardGaussian;
//    using Traits::Base::H;
//    using Traits::Base::StateDimension;

//public:
//    FactorizedLinearGaussianOservationModel2(
//            const Operator& noise_covariance,
//            const size_t observation_dimension = Observation::SizeAtCompileTime,
//            const size_t state_a_dimension = State_a::SizeAtCompileTime,
//            const size_t state_b_dimension = State_b::SizeAtCompileTime):
//        Traits::GaussianBase(observation_dimension),
//        state_a_dimension_(State_a::SizeAtCompileTime),
//        state_b_dimension_(State_b::SizeAtCompileTime),
//        H_a_(SensorMatrix_a::Zero(Dimension(), State_a_Dimension())),
//        H_b_(SensorMatrix_b::Zero(Dimension(), State_b_Dimension())),
//        Traits::Base(noise_covariance, )
//    {
//        Covariance(noise_covariance);
//    }


//protected:
//    virtual void Condition(const State_a& state_a,
//                           const State_b& state_b,
//                           size_t state_index,
//                           size_t pixel_index)
//    {
//        // delta_time_ = delta_time;

//        Mean(H_a_ * state_a + H_b_ * state_b);
//    }

//    virtual const SensorMatrix_a& H_a() const
//    {
//        return H_a_;
//    }

//    virtual const SensorMatrix_b& H_b() const
//    {
//        return H_b_;
//    }

//    virtual void H_a(const SensorMatrix_a& sensor_matrix_a)
//    {
//        H_a_ = sensor_matrix_a;
//    }

//    virtual void H_b(const SensorMatrix_b& sensor_matrix_b)
//    {
//        H_b_ = sensor_matrix_b;
//    }

//    virtual size_t State_a_Dimension() const
//    {
//        return state_a_dimension_;
//    }

//    virtual size_t State_b_Dimension() const
//    {
//        return state_b_dimension_;
//    }

//protected:
//    size_t state_a_dimension_;
//    size_t state_b_dimension_;
//    SensorMatrix_a H_a_;
//    SensorMatrix_b H_b_;
//};



}

#endif

