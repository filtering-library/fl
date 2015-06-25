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
    typedef FRBParticleFilter::Belief FRBParticleBelief;

    // gaussian filter
    typedef fl::GaussianFilter<ProcessModel, ObservationModel> GaussianFilter;
    typedef GaussianFilter::Belief GaussianBelief;


}
