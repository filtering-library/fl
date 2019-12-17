/*
 * This is part of the fl library, a C++ Bayesian filtering library
 * (https://github.com/filtering-library)
 *
 * Copyright (c) 2015 Max Planck Society,
 * 				 Autonomous Motion Department,
 * 			     Institute for Intelligent Systems
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
#include <fl/filter/gaussian/gaussian_filter_linear.hpp>
#include <fl/model/transition/linear_transition.hpp>
#include <fl/model/sensor/linear_gaussian_sensor.hpp>

template<typename Vector, typename Matrix>
bool moments_are_similar(Vector mean_a, Matrix cov_a,
                         Vector mean_b, Matrix cov_b, fl::Real epsilon = 0.1)
{
    Matrix cov_delta = cov_a.inverse() * cov_b;
    bool are_similar = cov_delta.isApprox(Matrix::Identity(), epsilon);

    Matrix square_root = fl::matrix_sqrt(cov_a);
    fl::Real max_mean_delta =
            (square_root.inverse() * (mean_a-mean_b)).cwiseAbs().maxCoeff();

    are_similar = are_similar && max_mean_delta < epsilon;

    return are_similar;
}

Eigen::Matrix<fl::Real, 3, 3> some_rotation()
{
    fl::Real angle = 2 * M_PI * fl::Real(rand()) / fl::Real(RAND_MAX);

    Eigen::Matrix<fl::Real, 3, 3> R = Eigen::Matrix<fl::Real, 3, 3>::Identity();

    R = R * Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitX());
    R = R * Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ());
    R = R * Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY());
    return R;
}

class ParticleFilterTest
    : public ::testing::Test
{
protected:

    typedef Eigen::Matrix<fl::Real, 3, 1> State;
    typedef Eigen::Matrix<fl::Real, 3, 1> Observation;
    typedef Eigen::Matrix<fl::Real, 3, 1> Input;

    typedef Eigen::Matrix<fl::Real, 3, 3> Matrix;

    typedef fl::LinearTransition<State, State, Input> Transition;
    typedef fl::LinearGaussianSensor<Observation, State> Sensor;

    // particle filter
    typedef fl::ParticleFilter<Transition, Sensor> ParticleFilter;
    typedef ParticleFilter::Belief ParticleBelief;

    // gaussian filter
    typedef fl::GaussianFilter<Transition, Sensor> GaussianFilter;
    typedef GaussianFilter::Belief GaussianBelief;

    ParticleFilterTest()
        : transition(create_transition()),
          sensor(create_sensor()),
          particle_filter(transition, sensor),
          gaussian_filter(transition, sensor)

    {
        N_particles = 10000;
        N_steps = 10;
        delta_time = 1;

        // create intial beliefs
        gaussian_belief.set_standard();
        particle_belief.from_distribution(gaussian_belief, N_particles);
    }

    Transition create_transition()
    {
        srand(0);

        Transition transition;

        transition.dynamics_matrix(some_rotation());

        Matrix R = some_rotation();
        Matrix D = Eigen::DiagonalMatrix<fl::Real, 3>(1, 3.5, 1.2);

        transition.noise_matrix(R*D);

        return transition;
    }

    Sensor create_sensor()
    {
        srand(0);

        Sensor sensor;

        sensor.sensor_matrix(some_rotation());

        Matrix R = some_rotation();
        Matrix D = Eigen::DiagonalMatrix<fl::Real, 3>(3.1, 1.0, 1.3);

        sensor.noise_covariance(R*D*R.transpose());

        return sensor;
    }

    Transition transition;
    Sensor sensor;

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
        particle_filter.predict(particle_belief, Input::Zero(), particle_belief);
        gaussian_filter.predict(gaussian_belief, Input::Zero(), gaussian_belief);

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

        particle_filter.update(particle_belief, observation, particle_belief);
        gaussian_filter.update(gaussian_belief, observation, gaussian_belief);

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
        state = transition.state(state,
                                    standard_gaussian.sample(),
                                    Input::Zero());
        Observation observation =
                sensor.observation(state, standard_gaussian.sample());

        // predict
        particle_filter.predict(particle_belief, State::Zero(), particle_belief);
        gaussian_filter.predict(gaussian_belief, State::Zero(), gaussian_belief);

        // update
        particle_filter.update(particle_belief, observation, particle_belief);
        gaussian_filter.update(gaussian_belief, observation, gaussian_belief);
    }

    State delta = particle_belief.mean() - gaussian_belief.mean();
    fl::Real mh_distance = delta.transpose() * gaussian_belief.precision() * delta;

    // make sure that the estimate of the pf is within one std dev
    EXPECT_TRUE(std::sqrt(mh_distance) <= 1.0);
}
