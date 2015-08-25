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
 * \file robust_gaussian_filter_plot_data.cpp
 * \date August 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <Eigen/Dense>

#include "../typecast.hpp"
#include <fl/util/meta.hpp>
#include <fl/filter/gaussian/gaussian_filter.hpp>
#include <fl/filter/gaussian/robust_gaussian_filter.hpp>
#include <fl/model/observation/linear_cauchy_observation_model.hpp>
#include <fl/model/observation/body_tail_observation_model.hpp>
#include <fl/model/observation/linear_gaussian_observation_model.hpp>

using namespace fl;

template <typename TestType>
struct RobutGaussianFilterPlotData
{
    typedef typename TestType::Parameter Configuration;

    enum: signed int
    {
        StateDim = Configuration::StateDim,
        InputDim = Configuration::InputDim,
        ObsrvDim = Configuration::ObsrvDim,

        StateSize = fl::TestSize<StateDim, TestType>::Value,
        InputSize = fl::TestSize<InputDim, TestType>::Value,
        ObsrvSize = fl::TestSize<ObsrvDim, TestType>::Value
    };

    typedef Eigen::Matrix<fl::Real, StateSize, 1> State;
    typedef Eigen::Matrix<fl::Real, InputSize, 1> Input;
    typedef Eigen::Matrix<fl::Real, ObsrvSize, 1> Obsrv;

    typedef fl::LinearStateTransitionModel<State, Input>     ProcessModel;
    typedef fl::LinearGaussianObservationModel<Obsrv, State> BodyModel;
    typedef fl::LinearCauchyObservationModel<Obsrv, State>   TailModel;
    typedef fl::BodyTailObsrvModel<BodyModel, TailModel >    BodyTailObsrvModel;

    ProcessModel create_linear_state_model()
    {
        auto model = ProcessModel(StateDim, InputDim);

        auto A = model.create_dynamics_matrix();
        auto Q = model.create_noise_matrix();

        Q.setIdentity();
        A.setIdentity();

        model.dynamics_matrix(A);
        model.noise_matrix(Q);

        return model;
    }

    BodyModel create_linear_gaussian_observation_model(Real covariance_factor)
    {
        auto model = BodyModel(ObsrvDim, StateDim);

        auto H = model.create_sensor_matrix();
        auto R = model.create_noise_matrix();

        H.setIdentity();
        R.setIdentity();
        R *= std::sqrt(covariance_factor);

        model.sensor_matrix(H);
        model.noise_matrix(R);

        return model;
    }

    TailModel create_linear_cauchy_observation_model(Real covariance_factor)
    {
        auto model = TailModel();

        auto R = model.noise_covariance();
        R.setIdentity();
        R *= covariance_factor;
        model.noise_covariance(R);

        return model;
    }

    template <typename BodyModel, typename TailModel>
    BodyTailObsrvModel create_body_tail_observation_model(
        BodyModel&& body_model,
        TailModel&& tail_model,
        Real weight)
    {

        auto model = BodyTailObsrvModel(body_model, tail_model, weight);
        return model;
    }

    typedef UnscentedQuadrature Quadrature;
//        typedef fl::SigmaPointQuadrature<
//                    fl::MonteCarloTransform<
//                        fl::ConstantPointCountPolicy<1000>>> Quadrature;

    typedef RobustGaussianFilter<
                ProcessModel,
                BodyTailObsrvModel,
                Quadrature
            > Filter;

    template <typename ProcessModel, typename ObsrvModel>
    Filter create_filter(ProcessModel&& process_model, ObsrvModel&& obsrv_model)
    {
        return Filter(process_model, obsrv_model, Quadrature());
    }

    State zero_state() { return State::Zero(StateDim); }
    Input zero_input() { return Input::Zero(InputDim); }
    Obsrv zero_obsrv() { return Obsrv::Zero(ObsrvDim); }

    State rand_state() { return State::Random(StateDim); }
    Input rand_input() { return Input::Random(InputDim); }
    Obsrv rand_obsrv() { return Obsrv::Random(ObsrvDim); }
};

template <int StateDimension, int InputDimension, int ObsrvDimension>
struct RobutGaussianFilterTestConfiguration
{
    enum : signed int
    {
        StateDim = StateDimension,
        InputDim = InputDimension,
        ObsrvDim = ObsrvDimension
    };
};

int main(int nargs, char** vargs)
{
    typedef RobutGaussianFilterTestConfiguration<1, 1, 1> Config;

    typedef typename RobutGaussianFilterPlotData<
                StaticTest<Config>
            >::Filter Filter;

    RobutGaussianFilterPlotData<StaticTest<Config>> rgf_pd;

    auto rgf = rgf_pd.create_filter(
                   rgf_pd.create_linear_state_model(),
                   rgf_pd.create_body_tail_observation_model(
                       rgf_pd.create_linear_gaussian_observation_model(1.0),
                       rgf_pd.create_linear_cauchy_observation_model(10.0),
                       0.1));

    auto belief = rgf.create_belief();

    rgf.predict(belief, rgf_pd.zero_input(), belief);
    rgf.update(belief, rgf_pd.rand_obsrv(), belief);

    PV(rgf.name());
    PV(rgf.description());
    PV(belief.mean());
    PV(belief.covariance());

    return 0;
}
