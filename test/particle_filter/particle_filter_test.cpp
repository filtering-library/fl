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


TEST(particle_filter, predict)
{  
    typedef Eigen::Matrix<double, 3, 1> State;
    typedef Eigen::Matrix<double, 3, 1> Observation;
    typedef Eigen::Matrix<double, 3, 1> Input;

    typedef Eigen::Matrix<double, 3, 3> Matrix;

    typedef fl::LinearGaussianProcessModel<State, Input> ProcessModel;
    typedef fl::LinearObservationModel<Observation, State> ObservationModel;

    typedef fl::LinearGaussianObservationModel<Observation, State>
                                                        OldObservationModel;


    // particle filter
    typedef fl::ParticleFilter<ProcessModel, ObservationModel> ParticleFilter;
    typedef ParticleFilter::StateDistribution ParticleBelief;

    // gaussian filter
    typedef fl::GaussianFilter<ProcessModel, OldObservationModel> GaussianFilter;
    typedef GaussianFilter::StateDistribution GaussianBelief;


    srand(0);
    size_t N_particles = 10000;
    size_t N_steps = 10;
    size_t delta_time = 1;


    // create process model
    ProcessModel process_model;
    {
        process_model.A(some_rotation());
        Matrix R = some_rotation();
        Matrix D = Eigen::DiagonalMatrix<double, 3>(1, 1.5, 3.2);
        process_model.covariance(R*D*R.transpose());
    }

    // create observation model
    /// \todo this is a hack because the GF does not currently work with the new
    /// observation model interface
    OldObservationModel old_observation_model;
    {
        old_observation_model.H(some_rotation());
        Matrix R = some_rotation();
        Matrix D = Eigen::DiagonalMatrix<double, 3>(1.1, 1.0, 3.3);
        old_observation_model.covariance(R*D*R.transpose());
    }
    ObservationModel observation_model;
    observation_model.sensor_matrix(old_observation_model.H());
    observation_model.noise_matrix(old_observation_model.square_root());

    // create filters
    ParticleFilter particle_filter(process_model, observation_model);
    GaussianFilter gaussian_filter(process_model, old_observation_model);

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

    typedef fl::LinearGaussianObservationModel<Observation, State>
                                                        OldObservationModel;


    // particle filter
    typedef fl::ParticleFilter<ProcessModel, ObservationModel> ParticleFilter;
    typedef ParticleFilter::StateDistribution ParticleBelief;

    // gaussian filter
    typedef fl::GaussianFilter<ProcessModel, OldObservationModel> GaussianFilter;
    typedef GaussianFilter::StateDistribution GaussianBelief;


    srand(0);
    size_t N_particles = 10000;
    size_t N_steps = 10;
    size_t delta_time = 1;


    // create process model
    ProcessModel process_model;
    {
        process_model.A(some_rotation());
        Matrix R = some_rotation();
        Matrix D = Eigen::DiagonalMatrix<double, 3>(1, 1.5, 3.2);
        process_model.covariance(R*D*R.transpose());
    }

    // create observation model
    /// \todo this is a hack because the GF does not currently work with the new
    /// observation model interface
    OldObservationModel old_observation_model;
    {
        old_observation_model.H(some_rotation());
        Matrix R = some_rotation();
        Matrix D = Eigen::DiagonalMatrix<double, 3>(1.1, 1.0, 3.3);
        old_observation_model.covariance(R*D*R.transpose());
    }

    // remove this
    old_observation_model.covariance(Eigen::Matrix3d::Identity());
    old_observation_model.H(Eigen::Matrix3d::Identity());



    ObservationModel observation_model;
    observation_model.sensor_matrix(old_observation_model.H());
    observation_model.noise_matrix(old_observation_model.square_root());

    // create filters
    ParticleFilter particle_filter(process_model, observation_model);
    GaussianFilter gaussian_filter(process_model, old_observation_model);

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

























