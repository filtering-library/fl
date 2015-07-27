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
 * \file linear_algebra_smw_inversion_test.cpp
 * \date 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <iostream>
#include <vector>
#include <ctime>

#include <fl/util/math.hpp>

typedef Eigen::Matrix<double, 3, 1> State;
typedef Eigen::Matrix<double, 1, 1> Observation;

const int INV_DIMENSION = 14;
const int SUBSAMPLING_FACTOR = 8;
const int OBSERVATION_DIMENSION = (640*480)/(SUBSAMPLING_FACTOR*SUBSAMPLING_FACTOR);
const int INVERSION_ITERATIONS = OBSERVATION_DIMENSION * 30;

TEST(InversionTests, SMWInversion)
{
    Eigen::MatrixXd cov = Eigen::MatrixXd::Random(15, 15);
    cov = cov * cov.transpose();

    Eigen::MatrixXd A = cov.block(0,   0, 14, 14);
    Eigen::MatrixXd B = cov.block(0,  14, 14,  1);
    Eigen::MatrixXd C = cov.block(14,  0, 1,  14);
    Eigen::MatrixXd D = cov.block(14, 14, 1,   1);

    Eigen::MatrixXd L_A;
    Eigen::MatrixXd L_B;
    Eigen::MatrixXd L_C;
    Eigen::MatrixXd L_D;

    Eigen::MatrixXd cov_inv = cov.inverse();
    Eigen::MatrixXd cov_smw_inv;

    Eigen::MatrixXd A_inv = A.inverse();
    fl::smw_inverse(A_inv, B, C, D, L_A, L_B, L_C, L_D, cov_smw_inv);

    EXPECT_TRUE(cov_smw_inv.isApprox(cov_inv));
}


TEST(InversionTests, SMWInversion_no_Lx)
{
    Eigen::MatrixXd cov = Eigen::MatrixXd::Random(15, 15);
    cov = cov * cov.transpose();

    Eigen::MatrixXd A = cov.block(0,   0, 14, 14);
    Eigen::MatrixXd B = cov.block(0,  14, 14,  1);
    Eigen::MatrixXd C = cov.block(14,  0, 1,  14);
    Eigen::MatrixXd D = cov.block(14, 14, 1,   1);

    Eigen::MatrixXd cov_inv = cov.inverse();
    Eigen::MatrixXd cov_smw_inv;

    Eigen::MatrixXd A_inv = A.inverse();
    fl::smw_inverse(A_inv, B, C, D, cov_smw_inv);

    EXPECT_TRUE(cov_smw_inv.isApprox(cov_inv));
}

// speed performance tests
//TEST(InversionTests, fullMatrixInversionSpeed)
//{
//    Eigen::MatrixXd cov = Eigen::MatrixXd::Random(INV_DIMENSION, INV_DIMENSION);
//    cov = cov * cov.transpose();

//    Eigen::MatrixXd cov_inv;

//    std::clock_t start = std::clock();
//    int number_of_inversions = 0;
//    while ( (( std::clock() - start ) / (double) CLOCKS_PER_SEC) < 1.0 )
//    {
//        cov_inv = cov.inverse();
//        number_of_inversions++;
//    }

////    std::cout << "fullMatrixInversionSpeed::number_of_inversions: "
////              << number_of_inversions
////              << "(" << number_of_inversions/OBSERVATION_DIMENSION << " fps)"
////              << std::endl;

//}

//TEST(InversionTests, SMWMatrixInversionSpeed)
//{
//    Eigen::MatrixXd cov = Eigen::MatrixXd::Random(INV_DIMENSION, INV_DIMENSION);
//    cov = cov * cov.transpose();

//    Eigen::MatrixXd A = cov.block(0, 0, INV_DIMENSION-1, INV_DIMENSION-1);
//    Eigen::MatrixXd B = cov.block(0, INV_DIMENSION-1, INV_DIMENSION-1, 1);
//    Eigen::MatrixXd C = cov.block(INV_DIMENSION-1, 0, 1, INV_DIMENSION-1);
//    Eigen::MatrixXd D = cov.block(INV_DIMENSION-1, INV_DIMENSION-1, 1, 1);
//    Eigen::MatrixXd A_inv = A.inverse();

//    Eigen::MatrixXd L_A = Eigen::MatrixXd(INV_DIMENSION-1, INV_DIMENSION-1);
//    Eigen::MatrixXd L_B = Eigen::MatrixXd(INV_DIMENSION-1, 1);
//    Eigen::MatrixXd L_C = Eigen::MatrixXd(1, INV_DIMENSION-1);
//    Eigen::MatrixXd L_D = Eigen::MatrixXd(1, 1);

//    Eigen::MatrixXd cov_smw_inv;
//    std::clock_t start = std::clock();
//    int number_of_inversions = 0;
//    while ( ((std::clock() - start) / (double) CLOCKS_PER_SEC) < 1.0 )
//    {
//        fl::smw_inverse(A_inv, B, C, D, L_A, L_B, L_C, L_D, cov_smw_inv);
//        number_of_inversions++;
//    }

////    std::cout << "SMWMatrixInversionSpeed::number_of_inversions: "
////              << number_of_inversions
////              << "(" << number_of_inversions/OBSERVATION_DIMENSION << " fps)"
////              << std::endl;
//}

//TEST(InversionTests, SMWBlockMatrixInversionSpeed)
//{
//    Eigen::MatrixXd cov = Eigen::MatrixXd::Random(INV_DIMENSION, INV_DIMENSION);
//    cov = cov * cov.transpose();

//    Eigen::MatrixXd A = cov.block(0, 0, INV_DIMENSION-1, INV_DIMENSION-1);
//    Eigen::MatrixXd B = cov.block(0, INV_DIMENSION-1, INV_DIMENSION-1, 1);
//    Eigen::MatrixXd C = cov.block(INV_DIMENSION-1, 0, 1, INV_DIMENSION-1);
//    Eigen::MatrixXd D = cov.block(INV_DIMENSION-1, INV_DIMENSION-1, 1, 1);
//    Eigen::MatrixXd A_inv = A.inverse();

//    Eigen::MatrixXd L_A = Eigen::MatrixXd(INV_DIMENSION-1, INV_DIMENSION-1);
//    Eigen::MatrixXd L_B = Eigen::MatrixXd(INV_DIMENSION-1, 1);
//    Eigen::MatrixXd L_C = Eigen::MatrixXd(1, INV_DIMENSION-1);
//    Eigen::MatrixXd L_D = Eigen::MatrixXd(1, 1);

//    std::clock_t start = std::clock();
//    int number_of_inversions = 0;
//    while ( (( std::clock() - start ) / (double) CLOCKS_PER_SEC) < 1.0 )
//    {
//        fl::smw_inverse(A_inv, B, C, D, L_A, L_B, L_C, L_D);
//        number_of_inversions++;
//    }

////    std::cout << "SMWMatrixBlockInversionSpeed::number_of_inversions: "
////              << number_of_inversions
////              << "(" << number_of_inversions/OBSERVATION_DIMENSION << " fps)"
////              << std::endl;
//}
