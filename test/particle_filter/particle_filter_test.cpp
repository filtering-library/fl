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

    std::cout << "cov " << cov_a << std::endl;
    Matrix square_root = fl::matrix_sqrt(cov_a);
    std::cout << "square_root" << square_root << std::endl;
    std::cout << "square_root * square_root.transpose() "
              << square_root * square_root.transpose() << std::endl;

    double max_mean_delta =
            (square_root.inverse() * (mean_a-mean_b)).cwiseAbs().maxCoeff();

    are_similar = are_similar && max_mean_delta < epsilon;

    return are_similar;
}





TEST(particle_filter, some_test)
{  
    typedef Eigen::Matrix<double, 3, 1> State;
    typedef Eigen::Matrix<double, 3, 1> Observation;
    typedef Eigen::Matrix<double, 3, 1> Input;

    typedef Eigen::Matrix<double, 3, 3> Matrix;

    typedef fl::LinearGaussianProcessModel<State, Input> ProcessModel;
    typedef fl::LinearGaussianObservationModel<Observation, State> ObservationModel;

    // particle filter
    typedef fl::ParticleFilter<ProcessModel, ObservationModel> ParticleFilter;
    typedef ParticleFilter::StateDistribution ParticleBelief;

    // gaussian filter
    typedef fl::GaussianFilter<ProcessModel, ObservationModel> GaussianFilter;
    typedef GaussianFilter::StateDistribution GaussianBelief;


    size_t N_particles = 1000;
    size_t N_steps = 10;
    size_t delta_time = 1;


    // create process model
    ProcessModel process_model;
    process_model.A(Matrix::Identity());
    process_model.covariance(Matrix::Identity());

    // create observation model
    ObservationModel observation_model;
    observation_model.H(Matrix::Identity());
    observation_model.covariance(Matrix::Identity());

    // create filters
    ParticleFilter particle_filter(process_model, observation_model);
    GaussianFilter gaussian_filter(process_model, observation_model);

    // create intial beliefs
    GaussianBelief gaussian_belief;
    gaussian_belief.mean(State::Zero());
    gaussian_belief.covariance(Matrix::Identity());

    ParticleBelief particle_belief;
    particle_belief.log_unnormalized_prob_mass(ParticleBelief::Function::Zero(N_particles));
    for(size_t i = 0; i < particle_belief.size(); i++)
    {
        particle_belief.location(i) = gaussian_belief.sample();
    }

    //
    ParticleBelief new_particle_belief;
    particle_filter.predict(delta_time, State::Zero(),
                            particle_belief, new_particle_belief);
    particle_belief = new_particle_belief;

    GaussianBelief new_gaussian_belief;
    gaussian_filter.predict(delta_time, State::Zero(),
                            gaussian_belief, new_gaussian_belief);
    gaussian_belief = new_gaussian_belief;

    EXPECT_TRUE(
    moments_are_similar(particle_belief.mean(), particle_belief.covariance(),
                        gaussian_belief.mean(), gaussian_belief.covariance()));








}

