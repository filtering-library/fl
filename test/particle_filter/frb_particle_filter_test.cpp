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

#include <fl/filter/particle/frb_particle_filter.hpp>
#include <fl/filter/gaussian/gaussian_filter_kf.hpp>
#include <fl/model/observation/linear_observation_model.hpp>
#include <fl/model/process/linear_process_model.hpp>
#include <fl/model/process/binary_transition_density.hpp>

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


TEST(frb_particle_filter, predict)
{
    typedef Eigen::Matrix<double, 3, 1> State;
    typedef Eigen::Matrix<double, 3, 1> Observation;
    typedef Eigen::Matrix<double, 3, 1> Input;

    typedef Eigen::Matrix<double, 3, 3> Matrix;

    typedef fl::LinearGaussianProcessModel<State, Input> ProcessModel;
    typedef fl::BinaryTransitionDensity SwitchingDensity;
    typedef fl::LinearObservationModel<Observation, State> ObservationModel;

    // particle filter
    typedef fl::FRBParticleFilter<ProcessModel,
                                  SwitchingDensity,
                                  ObservationModel> FRBParticleFilter;
    typedef FRBParticleFilter::StateDistribution FRBParticleBelief;

    // gaussian filter
    typedef fl::GaussianFilter<ProcessModel, ObservationModel> GaussianFilter;
    typedef GaussianFilter::StateDistribution GaussianBelief;


}





































