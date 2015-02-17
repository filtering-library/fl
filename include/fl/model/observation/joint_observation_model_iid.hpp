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
 * \file joint_observation_model_iid.hpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__MODEL__OBSERVATION__JOINT_OBSERVATION_MODEL_IID_HPP
#define FL__MODEL__OBSERVATION__JOINT_OBSERVATION_MODEL_IID_HPP

#include <Eigen/Dense>

#include <memory>

#include <fl/util/traits.hpp>
#include <fl/util/meta.hpp>
#include <fl/distribution/gaussian.hpp>

#include <fl/model/observation/observation_model_interface.hpp>

namespace fl
{

// Forward declarations
template <typename...Model> class JointObservationModel;

/**
 * Traits of JointObservationModel<MultipleOf<ObservationModel, Count>>
 */
template <
    typename ObservationModel,
    int Count
>
struct Traits<
           JointObservationModel<MultipleOf<ObservationModel, Count>>
        >
{
    enum : signed int { ModelCount = Count };

    typedef ObservationModel LocalObservationModel;
    typedef typename Traits<ObservationModel>::Scalar Scalar;
    typedef typename Traits<ObservationModel>::State LocalState;
    typedef typename Traits<ObservationModel>::Noise LocalNoise;
    typedef typename Traits<ObservationModel>::Observation LocalObsrv;

    enum : signed int
    {
        ObsrvDim = ExpandSizes<LocalObsrv::SizeAtCompileTime, Count>::Size,
        StateDim = ExpandSizes<LocalState::SizeAtCompileTime, Count>::Size,
        NoiseDim = ExpandSizes<LocalNoise::SizeAtCompileTime, Count>::Size,
    };

    typedef Eigen::Matrix<Scalar, ObsrvDim, 1> Observation;
    typedef Eigen::Matrix<Scalar, StateDim, 1> State;
    typedef Eigen::Matrix<Scalar, NoiseDim, 1> Noise;

    typedef ObservationModelInterface<
                Observation,
                State,
                Noise
            > ObservationModelBase;
};

/**
 * \ingroup observation_models
 */
template <
    typename LocalObservationModel,
    int Count>
class JointObservationModel<MultipleOf<LocalObservationModel, Count>>
    : public Traits<
                 JointObservationModel<MultipleOf<LocalObservationModel, Count>>
             >::ObservationModelBase
{
public:
    typedef JointObservationModel<MultipleOf<LocalObservationModel,Count>> This;

    typedef typename Traits<This>::Observation Observation;
    typedef typename Traits<This>::State State;
    typedef typename Traits<This>::Noise Noise;

public:
    explicit JointObservationModel(
            const std::shared_ptr<LocalObservationModel>& local_obsrv_model,
            int count = ToDimension<Count>::Value)
        : local_obsrv_model_(local_obsrv_model),
          count_(count)
    {
        assert(count > 0);
    }

    /**
     * \brief Overridable default destructor
     */
    ~JointObservationModel() { }

    /**
     * \copydoc ObservationModelInterface::predict_observation
     */
    virtual Observation predict_observation(const State& state,
                                            const Noise& noise,
                                            double delta_time)
    {
        Observation y = Observation::Zero(observation_dimension(), 1);

        int obsrv_dim = local_obsrv_model_->observation_dimension();
        int noise_dim = local_obsrv_model_->noise_dimension();
        int state_dim = local_obsrv_model_->state_dimension();

        const int count = count_;
        for (int i = 0; i < count; ++i)
        {
            y.middleRows(i * obsrv_dim, obsrv_dim) =
                local_obsrv_model_->predict_observation(
                    state.middleRows(i * state_dim, state_dim),
                    noise.middleRows(i * noise_dim, noise_dim),
                    delta_time);
        }

        return y;
    }

    virtual size_t observation_dimension() const
    {
        return local_obsrv_model_->observation_dimension() * count_;
    }

    virtual size_t state_dimension() const
    {
        return local_obsrv_model_->state_dimension() * count_;
    }

    virtual size_t noise_dimension() const
    {
        return local_obsrv_model_->noise_dimension() * count_;
    }

    const std::shared_ptr<LocalObservationModel>& local_observation_model()
    {
        return local_obsrv_model_;
    }

protected:
    std::shared_ptr<LocalObservationModel> local_obsrv_model_;
    int count_;
};

}

#endif

