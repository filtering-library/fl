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
 * @date 2015
 * @author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * Max-Planck-Institute for Intelligent Systems
 */

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <fl/util/math/linear_algebra.hpp>

#include <fl/filter/particle/particle_filter.hpp>
#include <fl/filter/gaussian/gaussian_filter_kf.hpp>
#include <fl/model/sensor/linear_sensor.hpp>
#include <fl/model/process/linear_transition.hpp>

template<typename Vector, typename Matrix>
bool moments_are_similar(Vector mean_a, Matrix cov_a,
                         Vector mean_b, Matrix cov_b, double epsilon = 0.1)
{
    Matrix cov_delta = cov_a.inverse() * cov_b;
    bool are_similar = cov_delta.isApprox(Matrix::Identity(), epsilon);

    Matrix square_root = fl::matrix_sqrt(cov_a);
    double max_mean_delta =
            (square_root.inverse() * (mean_a-mean_b)).cwiseAbs().maxCoeff();

    are_similar = are_similar && max_mean_delta < epsilon;

    return are_similar;
}

Eigen::Matrix<double, 3, 3> some_rotation()
{
    double angle = 2 * M_PI * double(rand()) / double(RAND_MAX);

    Eigen::Matrix<double, 3, 3> R = Eigen::Matrix<double, 3, 3>::Identity();

    R = R * Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitX());
    R = R * Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ());
    R = R * Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY());
    return R;
}

TEST(particle_filter, predict)
{
    typedef Eigen::Matrix<double, 3, 1> State;
    typedef Eigen::Matrix<double, 3, 1> Observation;
    typedef Eigen::Matrix<double, 3, 1> Input;

    typedef Eigen::Matrix<double, 3, 3> Matrix;

    typedef fl::LinearGaussianProcessModel<State, Input> ProcessModel;
    typedef fl::LinearObservationModel<Observation, State> ObservationModel;

    // particle filter
    typedef fl::ParticleFilter<ProcessModel, ObservationModel> ParticleFilter;
    typedef ParticleFilter::Belief ParticleBelief;

    // gaussian filter
    typedef fl::GaussianFilter<ProcessModel, ObservationModel> GaussianFilter;
    typedef GaussianFilter::Belief GaussianBelief;


    srand(0);
    size_t N_particles = 10000;
    size_t N_steps = 10;
    size_t delta_time = 1;


    // create process model
    ProcessModel transition;
    {
        transition.A(some_rotation());
        Matrix R = some_rotation();
        Matrix D = Eigen::DiagonalMatrix<double, 3>(1, 3.5, 1.2);
        transition.covariance(R*D*R.transpose());
    }

    // create observation model
    /// \todo this is a hack because the GF does not currently work with the new
    /// observation model interface
    ObservationModel sensor;
    {
        sensor.sensor_matrix(some_rotation());
        Matrix R = some_rotation();
        Matrix D = Eigen::DiagonalMatrix<double, 3>(3.1, 1.0, 1.3);
        D = D.cwiseSqrt();
        sensor.noise_matrix(R*D);
    }

    // create filters
    ParticleFilter particle_filter(transition, sensor);
    GaussianFilter gaussian_filter(transition, sensor);

    // create intial beliefs
    GaussianBelief gaussian_belief;
    {
        gaussian_belief.mean(State::Zero());
        gaussian_belief.covariance(Matrix::Identity());
    }
    ParticleBelief particle_belief;
    particle_belief.from_distribution(gaussian_belief, N_particles);

    // run prediction
    for(size_t i = 0; i < N_steps; i++)
    {
        particle_filter.predict(delta_time, State::Zero(),
                                particle_belief, particle_belief);

        gaussian_filter.predict(delta_time, State::Zero(),
                                gaussian_belief, gaussian_belief);

        EXPECT_TRUE(moments_are_similar(
                        particle_belief.mean(), particle_belief.covariance(),
                        gaussian_belief.mean(), gaussian_belief.covariance()));
    }
}

TEST(particle_filter, update)
{
    typedef Eigen::Matrix<double, 3, 1> State;
    typedef Eigen::Matrix<double, 3, 1> Observation;
    typedef Eigen::Matrix<double, 3, 1> Input;

    typedef Eigen::Matrix<double, 3, 3> Matrix;

    typedef fl::LinearGaussianProcessModel<State, Input> ProcessModel;
    typedef fl::LinearObservationModel<Observation, State> ObservationModel;
    // particle filter
    typedef fl::ParticleFilter<ProcessModel, ObservationModel> ParticleFilter;
    typedef ParticleFilter::Belief ParticleBelief;

    // gaussian filter
    typedef fl::GaussianFilter<ProcessModel, ObservationModel> GaussianFilter;
    typedef GaussianFilter::Belief GaussianBelief;


    srand(0);
    size_t N_particles = 10000;
    size_t N_steps = 10;


    // create process model
    ProcessModel transition;
    {
        transition.A(some_rotation());
        Matrix R = some_rotation();
        Matrix D = Eigen::DiagonalMatrix<double, 3>(1, 3.5, 1.2);
        transition.covariance(R*D*R.transpose());
    }

    // create observation model
    /// \todo this is a hack because the GF does not currently work with the new
    /// observation model interface
    ObservationModel sensor;
    {
        sensor.sensor_matrix(some_rotation());
        Matrix R = some_rotation();
        Matrix D = Eigen::DiagonalMatrix<double, 3>(3.1, 1.0, 1.3);
        D = D.cwiseSqrt();
        sensor.noise_matrix(R*D);
    }

    // create filters
    ParticleFilter particle_filter(transition, sensor);
    GaussianFilter gaussian_filter(transition, sensor);

    // create intial beliefs
    GaussianBelief gaussian_belief;
    {
        gaussian_belief.mean(State::Zero());
        gaussian_belief.covariance(Matrix::Identity());
    }
    ParticleBelief particle_belief;
    particle_belief.from_distribution(gaussian_belief, N_particles);


    // run prediction
    for(size_t i = 0; i < N_steps; i++)
    {
        Observation observation(0.5, 0.5, 0.5);

        particle_filter.update(observation, particle_belief, particle_belief);
        gaussian_filter.update(observation, gaussian_belief, gaussian_belief);

        EXPECT_TRUE(moments_are_similar(
                        particle_belief.mean(), particle_belief.covariance(),
                        gaussian_belief.mean(), gaussian_belief.covariance()));
    }

}

TEST(particle_filter, predict_and_update)
{
    typedef Eigen::Matrix<double, 3, 1> State;
    typedef Eigen::Matrix<double, 3, 1> Observation;
    typedef Eigen::Matrix<double, 3, 1> Input;

    typedef Eigen::Matrix<double, 3, 3> Matrix;

    typedef fl::LinearGaussianProcessModel<State, Input> ProcessModel;
    typedef fl::LinearObservationModel<Observation, State> ObservationModel;

    // particle filter
    typedef fl::ParticleFilter<ProcessModel, ObservationModel> ParticleFilter;
    typedef ParticleFilter::Belief ParticleBelief;

    // gaussian filter
    typedef fl::GaussianFilter<ProcessModel, ObservationModel> GaussianFilter;
    typedef GaussianFilter::Belief GaussianBelief;


    srand(0);
    size_t N_particles = 10000;
    size_t N_steps = 10;
    size_t delta_time = 1;


    // create process model
    ProcessModel transition;
    {
        transition.A(some_rotation());
        Matrix R = some_rotation();
        Matrix D = Eigen::DiagonalMatrix<double, 3>(1, 3.5, 1.2);
        transition.covariance(R*D*R.transpose());
    }

    // create observation model
    /// \todo this is a hack because the GF does not currently work with the new
    /// observation model interface
    ObservationModel sensor;
    {
        sensor.sensor_matrix(some_rotation());
        Matrix R = some_rotation();
        Matrix D = Eigen::DiagonalMatrix<double, 3>(3.1, 1.0, 1.3);
        D = D.cwiseSqrt();
        sensor.noise_matrix(R*D);
    }

    // create filters
    ParticleFilter particle_filter(transition, sensor);
    GaussianFilter gaussian_filter(transition, sensor);

    // create intial beliefs
    GaussianBelief gaussian_belief;
    {
        gaussian_belief.mean(State::Zero());
        gaussian_belief.covariance(Matrix::Identity());
    }
    ParticleBelief particle_belief;
    particle_belief.from_distribution(gaussian_belief, N_particles);


    fl::StandardGaussian<State> standard_gaussian;
    State state = gaussian_belief.sample();
    // run prediction
    for(size_t i = 0; i < N_steps; i++)
    {
        // simulate system
        state = transition.predict_state(delta_time,
                                            state,
                                            standard_gaussian.sample(),
                                            State::Zero());
        Observation observation =
                sensor.observation(state, standard_gaussian.sample());

        // predict
        particle_filter.predict(delta_time, State::Zero(),
                                particle_belief, particle_belief);
        gaussian_filter.predict(delta_time, State::Zero(),
                                gaussian_belief, gaussian_belief);

        // update
        particle_filter.update(observation, particle_belief, particle_belief);
        gaussian_filter.update(observation, gaussian_belief, gaussian_belief);
    }

    State delta = particle_belief.mean() - gaussian_belief.mean();
    double mh_distance = delta.transpose() * gaussian_belief.precision() * delta;

    // make sure that the estimate of the pf is within one std dev
    EXPECT_TRUE(std::sqrt(mh_distance) <= 1.0);
}
