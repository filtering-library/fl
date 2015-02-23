/*
 * This is part of the FL library, a C++ Bayesian filtering library
 * (https://github.com/filtering-library)
 *
 * Copyright (c) 2014 Jan Issac (jan.issac@gmail.com)
 * Copyright (c) 2014 Manuel Wuthrich (manuel.wuthrich@gmail.com)
 *
 * Max-Planck Institute for Intelligent Systems, AMD Lab
 * University of Southern California, CLMC Lab
 *
 * This Source Code Form is subject to the terms of the MIT License (MIT).
 * A copy of the license can be found in the LICENSE file distributed with this
 * source code.
 */

/**
 * \file linear_observation_model.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__MODEL__OBSERVATION__LINEAR_OBSERVATION_MODEL_HPP
#define FL__MODEL__OBSERVATION__LINEAR_OBSERVATION_MODEL_HPP

#include <fl/util/traits.hpp>
#include <fl/distribution/gaussian.hpp>
#include <fl/model/observation/observation_model_interface.hpp>

namespace fl
{

// Forward declarations
template <
    typename Observation,
    typename State>
class LinearGaussianObservationModel;


/**
 * Linear Gaussian observation model traits. This trait definition contains all
 * types used internally within the model. Additionally, it provides the types
 * needed externally to use the linear Gaussian model.
 */
template <
    typename Observation_,
    typename State_>
struct Traits<
           LinearGaussianObservationModel<Observation_, State_>>
{    
    typedef State_ State;
    typedef Observation_ Observation;

    typedef Gaussian<Observation> GaussianBase;
    typedef typename Traits<GaussianBase>::Scalar Scalar;
    typedef typename Traits<GaussianBase>::SecondMoment SecondMoment;
    typedef typename Traits<GaussianBase>::StandardVariate Noise;

    typedef Eigen::Matrix<
                Scalar,
                Observation::SizeAtCompileTime,
                State::SizeAtCompileTime
            > SensorMatrix;

    typedef ObservationModelInterface<
                Observation,
                State,
                Noise
            > ObservationModelBase;
};

/**
 * \ingroup observation_models
 */
template <typename Observation,typename State>
class LinearGaussianObservationModel
    : public Traits<
                 LinearGaussianObservationModel<Observation, State>
             >::ObservationModelBase,
      public Traits<
                 LinearGaussianObservationModel<Observation, State>
             >::GaussianBase
{
public:
    typedef LinearGaussianObservationModel<Observation, State> This;

    typedef typename Traits<This>::Noise Noise;
    typedef typename Traits<This>::Scalar Scalar;
    typedef typename Traits<This>::SecondMoment SecondMoment;
    typedef typename Traits<This>::SensorMatrix SensorMatrix;

    using Traits<This>::GaussianBase::mean;
    using Traits<This>::GaussianBase::covariance;
    using Traits<This>::GaussianBase::dimension;

public:
    LinearGaussianObservationModel(
            const SecondMoment& noise_covariance,
            const int observation_dim = DimensionOf<Observation>(),
            const int state_dim = DimensionOf<State>())
        : Traits<This>::GaussianBase(observation_dim),
          state_dimension_(state_dim),
          H_(SensorMatrix::Ones(observation_dimension(),
                                state_dimension()))
    {
        covariance(noise_covariance);
    }

    ~LinearGaussianObservationModel() { }

    virtual void condition(const State& x)
    {
        mean(H_ * x);
    }

    virtual const SensorMatrix& H() const
    {
        return H_;
    }

    virtual void H(const SensorMatrix& sensor_matrix)
    {
        H_ = sensor_matrix;
    }

    virtual Observation predict_observation(const State& state,
                                            const Noise& noise,
                                            double delta_time)
    {
        condition(state);
        return Traits<This>::GaussianBase::map_standard_normal(noise);
    }

    virtual int observation_dimension() const
    {
        return Traits<This>::GaussianBase::dimension();
    }

    virtual int noise_dimension() const
    {
        return Traits<This>::GaussianBase::standard_variate_dimension();
    }

    virtual int state_dimension() const
    {
        return state_dimension_;
    }


protected:
    int state_dimension_;
    SensorMatrix H_;
};


//// Forward declarations
//template <typename Observation_, typename State_a_, typename State_b_>
//class FactorizedLinearGaussianObservationModel;

//template <typename Observation_, typename State_a_, typename State_b_>
//struct Traits<FactorizedLinearGaussianObservationModel<Observation_,
//                                                      State_a_,
//                                                      State_b_>>
//{
//    typedef Observation_ Observation;
//    typedef State_a_ State_a;
//    typedef State_b_ State_b;
//    typedef Gaussian<Observation_> GaussianBase;
//    typedef typename Traits<GaussianBase>::Scalar Scalar;
//    typedef typename Traits<GaussianBase>::SecondMoment SecondMoment;
//    typedef typename Traits<GaussianBase>::StandardVariate Noise;
//    typedef Eigen::Matrix<Scalar,
//                          Observation::SizeAtCompileTime,
//                          State_a::SizeAtCompileTime> SensorMatrix_a;
//    typedef Eigen::Matrix<Scalar,
//                          Observation::SizeAtCompileTime,
//                          State_b::SizeAtCompileTime> SensorMatrix_b;
//};

//template <typename Observation_,typename State_a_, typename State_b_>
//class FactorizedLinearGaussianObservationModel:
//    public Traits<
//               FactorizedLinearGaussianObservationModel<
//                   Observation_, State_a_, State_b_>>::GaussianBase
//{
//public:
//    typedef FactorizedLinearGaussianObservationModel<
//                Observation_,
//                State_a_,
//                State_b_
//            > This;

//    typedef typename Traits<This>::Noise Noise;
//    typedef typename Traits<This>::Scalar Scalar;
//    typedef typename Traits<This>::SecondMoment SecondMoment;
//    typedef typename Traits<This>::Observation Observation;
//    typedef typename Traits<This>::State_a State_a;
//    typedef typename Traits<This>::State_b State_b;
//    typedef typename Traits<This>::SensorMatrix_a SensorMatrix_a;
//    typedef typename Traits<This>::SensorMatrix_b SensorMatrix_b;

//    using Traits<This>::GaussianBase::mean;
//    using Traits<This>::GaussianBase::covariance;
//    using Traits<This>::GaussianBase::dimension;

//public:
//    FactorizedLinearGaussianObservationModel(
//            const SecondMoment& noise_covariance,
//            const size_t observation_dimension = Observation::SizeAtCompileTime,
//            const size_t state_a_dimension = State_a::SizeAtCompileTime,
//            const size_t state_b_dimension = State_b::SizeAtCompileTime):
//        Traits<This>::GaussianBase(observation_dimension),
//        state_a_dimension_(state_a_dimension),
//        state_b_dimension_(state_b_dimension),
//        H_a_(SensorMatrix_a::Zero(dimension(), State_a_dimension())),
//        H_b_(SensorMatrix_b::Zero(dimension(), State_b_dimension()))
//    {
//        covariance(noise_covariance);
//    }

//    ~FactorizedLinearGaussianObservationModel() { }

//    virtual Observation Predict(const Noise& noise) const
//    {
//        return map_standard_normal(noise);
//    }

//    virtual void condition(const State_a& state_a,
//                           const State_b& state_b,
//                           size_t state_index,
//                           size_t pixel_index)
//    {
//        mean(H_a_ * state_a + H_b_ * state_b);
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

//    virtual size_t State_a_dimension() const
//    {
//        return state_a_dimension_;
//    }

//    virtual size_t State_b_dimension() const
//    {
//        return state_b_dimension_;
//    }

//protected:
//    size_t state_a_dimension_;
//    size_t state_b_dimension_;
//    SensorMatrix_a H_a_;
//    SensorMatrix_b H_b_;
//};





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
//    typedef typename Traits<GaussianBase>::SecondMoment SecondMoment;
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
//    typedef FactorizedLinearGaussianOservationModel2<Observation_, State_a_, State_b_> Traits;

//    typedef typename Traits<This>::State_a State_a;
//    typedef typename Traits<This>::State_b State_b;
//    typedef typename Traits<This>::Observation Observation;
//    typedef typename Traits<This>::Noise Noise;
//    typedef typename Traits<This>::Scalar Scalar;
//    typedef typename Traits<This>::SecondMoment SecondMoment;
//    typedef typename Traits<This>::SensorMatrix_a SensorMatrix_a;
//    typedef typename Traits<This>::SensorMatrix_b SensorMatrix_b;

//    using Traits<This>::GaussianBase::Mean;
//    using Traits<This>::GaussianBase::Covariance;
//    using Traits<This>::GaussianBase::Dimension;
//    using Traits<This>::Base::map_standard_normal;
//    using Traits<This>::Base::H;
//    using Traits<This>::Base::StateDimension;

//public:
//    FactorizedLinearGaussianOservationModel2(
//            const SecondMoment& noise_covariance,
//            const size_t observation_dimension = Observation::SizeAtCompileTime,
//            const size_t state_a_dimension = State_a::SizeAtCompileTime,
//            const size_t state_b_dimension = State_b::SizeAtCompileTime):
//        Traits<This>::GaussianBase(observation_dimension),
//        state_a_dimension_(State_a::SizeAtCompileTime),
//        state_b_dimension_(State_b::SizeAtCompileTime),
//        H_a_(SensorMatrix_a::Zero(dimension(), State_a_dimension())),
//        H_b_(SensorMatrix_b::Zero(dimension(), State_b_dimension())),
//        Traits<This>::Base(noise_covariance, )
//    {
//        covariance(noise_covariance);
//    }


//protected:
//    virtual void condition(const State_a& state_a,
//                           const State_b& state_b,
//                           size_t state_index,
//                           size_t pixel_index)
//    {
//        // delta_time_ = delta_time;

//        mean(H_a_ * state_a + H_b_ * state_b);
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

//    virtual size_t State_a_dimension() const
//    {
//        return state_a_dimension_;
//    }

//    virtual size_t State_b_dimension() const
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

