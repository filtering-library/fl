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
#include <fl/model/observation/linear_observation_model.hpp>
#include <fl/model/process/linear_process_model.hpp>

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

class ParticleFilterTest
    : public ::testing::Test
{
protected:

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

    ParticleFilterTest()
        : process_model(create_process_model()),
          observation_model(create_observation_model()),
          particle_filter(process_model, observation_model),
          gaussian_filter(process_model, observation_model)

    {
        N_particles = 10000;
        N_steps = 10;
        delta_time = 1;

        // create intial beliefs
        gaussian_belief.set_standard();
        particle_belief.from_distribution(gaussian_belief, N_particles);
    }

    ProcessModel create_process_model()
    {
        srand(0);

        ProcessModel process_model;

        process_model.A(some_rotation());
        Matrix R = some_rotation();
        Matrix D = Eigen::DiagonalMatrix<double, 3>(1, 3.5, 1.2);
        process_model.covariance(R*D*R.transpose());

        return process_model;
    }

    ObservationModel create_observation_model()
    {
        srand(0);

        ObservationModel observation_model;

        observation_model.sensor_matrix(some_rotation());
        Matrix R = some_rotation();
        Matrix D = Eigen::DiagonalMatrix<double, 3>(3.1, 1.0, 1.3);
        D = D.cwiseSqrt();
        observation_model.noise_matrix(R*D);

        return observation_model;
    }

    ProcessModel process_model;
    ObservationModel observation_model;

    ParticleFilter particle_filter;
    GaussianFilter gaussian_filter;

    GaussianBelief gaussian_belief;
    ParticleBelief particle_belief;

    size_t N_particles;
    size_t N_steps;
    size_t delta_time;

};

TEST_F(ParticleFilterTest, predict)
{
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

TEST_F(ParticleFilterTest, update)
{
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

TEST_F(ParticleFilterTest, predict_and_update)
{
    fl::StandardGaussian<State> standard_gaussian;
    State state = gaussian_belief.sample();
    // run prediction
    for(size_t i = 0; i < N_steps; i++)
    {
        // simulate system
        state = process_model.predict_state(delta_time,
                                            state,
                                            standard_gaussian.sample(),
                                            State::Zero());
        Observation observation =
                observation_model.observation(state, standard_gaussian.sample());

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
