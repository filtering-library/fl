

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
 * @date 2014
 * @author Jan Issac (jan.issac@gmail.com)
 * Max-Planck-Institute for Intelligent Systems,
 * University of Southern California
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <iostream>

#include <fl/distribution/gaussian.hpp>

class GaussianTests:
        public testing::Test
{
public:
    typedef Eigen::Matrix<double, 5, 1> FVector;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> DVector;

protected:
    template <typename Gaussian>
    void test_gaussian_dimension(Gaussian& gaussian, size_t dim)
    {
        EXPECT_EQ(gaussian.Dimension(), dim);
        EXPECT_EQ(gaussian.NoiseDimension(), dim);
        EXPECT_EQ(gaussian.Mean().rows(), dim);
        EXPECT_EQ(gaussian.Covariance().rows(), dim);
        EXPECT_EQ(gaussian.Covariance().cols(), dim);
        EXPECT_EQ(gaussian.Precision().rows(), dim);
        EXPECT_EQ(gaussian.Precision().cols(), dim);
        EXPECT_EQ(gaussian.SquareRoot().rows(), dim);
        EXPECT_EQ(gaussian.SquareRoot().cols(), dim);

        typename Gaussian::Noise noise =
                Gaussian::Noise::Random(gaussian.NoiseDimension(),1);
        EXPECT_EQ(gaussian.MapStandardGaussian(noise).rows(), dim);
    }

    template <typename Gaussian>
    void test_gaussian_covariance(Gaussian& gaussian)
    {
        /*
        typedef typename fl::Traits<Gaussian>::Operator Covariance;

        Covariance covariance = Eigen::MatrixXd::Identity(
                                    gaussian.Dimension(),
                                    gaussian.Dimension());
        Covariance square_root = covariance;
        Covariance precision = covariance;

        // first verify standard gaussian
        {
            SCOPED_TRACE("Unchanged");

            test_gaussian_attributes(
                        gaussian, covariance, precision, square_root);
        }

        // set covariance and verify representations
        {
            SCOPED_TRACE("Covariance setter");

            covariance.setRandom();
            covariance *= covariance.transpose();
            square_root = covariance.llt().matrixL();
            precision = covariance.inverse();
            gaussian.Covariance(covariance);
            test_gaussian_attributes(
                        gaussian, covariance, precision, square_root);
        }

        // set square root and verify representations
        {
            SCOPED_TRACE("SquareRoot setter");

            square_root.setRandom();
            covariance = square_root * square_root.transpose();
            precision = covariance.inverse();
            gaussian.SquareRoot(square_root);
            test_gaussian_attributes(
                        gaussian, covariance, precision, square_root);
        }

        // set covariance and verify representations
        {
            SCOPED_TRACE("Precision setter");

            precision.setRandom();
            precision *= precision.transpose();
            covariance= precision .inverse();
            square_root = covariance.llt().matrixL();
            gaussian.Precision(precision);
            test_gaussian_attributes(
                        gaussian, covariance, precision, square_root);
        }
        */
    }

    template <typename Gaussian, typename Covariance>
    void test_gaussian_attributes(Gaussian& gaussian,
                                  const Covariance& covariance,
                                  const Covariance& precision,
                                  const Covariance& square_root)
    {
        EXPECT_GT(gaussian.Dimension(), 0);

        EXPECT_TRUE(gaussian.Covariance().isApprox(covariance));
        EXPECT_TRUE(gaussian.Precision().isApprox(precision));
        const Covariance temp =
                gaussian.SquareRoot() * gaussian.SquareRoot().transpose();
        const Covariance temp2 = square_root * square_root.transpose();
        //EXPECT_TRUE(temp.isApprox(temp2));
        EXPECT_TRUE(gaussian.HasFullRank());
    }
};

//TEST_F(GaussianTests, eigen_O3_isApprox_bug)
//{
//    Eigen::MatrixXd m = Eigen::MatrixXd::Random(5, 5);
//    bool expect_false = (m * m).isApprox(m * m.transpose());
//}


TEST_F(GaussianTests, fixed_dimension)
{
    typedef Eigen::Matrix<double, 10, 1> Vector;
    fl::Gaussian<Vector> gaussian;

    test_gaussian_dimension(gaussian, 10);
}

TEST_F(GaussianTests, dynamic_dimension)
{
    const size_t dim = 10;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
    fl::Gaussian<Vector> gaussian(dim);

    test_gaussian_dimension(gaussian, dim);
}

TEST_F(GaussianTests, fixed_standard_covariance)
{
    typedef fl::Gaussian<FVector> Gaussian;
    typedef typename fl::Traits<Gaussian>::Operator Covariance;

    Gaussian gaussian;
    Covariance covariance = Eigen::MatrixXd::Identity(
                                gaussian.Dimension(),
                                gaussian.Dimension());

    test_gaussian_attributes(gaussian, covariance, covariance, covariance);
    gaussian.SetStandard();
    test_gaussian_attributes(gaussian, covariance, covariance, covariance);
    // gaussian.SetStandard(10); // causes compile time error as expected
}


TEST_F(GaussianTests, dynamic_standard_covariance)
{
    typedef fl::Gaussian<DVector> Gaussian;
    typedef typename fl::Traits<Gaussian>::Operator Covariance;

    Gaussian gaussian(6);
    Covariance covariance = Eigen::MatrixXd::Identity(gaussian.Dimension(),
                                                      gaussian.Dimension());

    {
        SCOPED_TRACE("Unchanged");

        EXPECT_EQ(gaussian.Dimension(), 6);
        test_gaussian_attributes(gaussian, covariance, covariance, covariance);
    }

    {
        SCOPED_TRACE("gaussian.SetStandard()");

        gaussian.SetStandard();
        EXPECT_EQ(gaussian.Dimension(), 6);
        //test_gaussian_attributes(gaussian, covariance, covariance, covariance);
    }

    {
        SCOPED_TRACE("gaussian.SetStandard(10)");

        gaussian.Dimension(10);
        EXPECT_EQ(gaussian.Dimension(), 10);
        covariance = Eigen::MatrixXd::Identity(gaussian.Dimension(),
                                               gaussian.Dimension());
        //test_gaussian_attributes(gaussian, covariance, covariance, covariance);
    }
}

TEST_F(GaussianTests, fixed_gaussian_covariance)
{
    // triggers static assert as expected:
    // fl::Gaussian<Eigen::Matrix<double, 0, 0>> gaussian;

    fl::Gaussian<FVector> gaussian;
    EXPECT_NO_THROW(test_gaussian_covariance(gaussian));
}

TEST_F(GaussianTests, dynamic_gaussian_covariance_constructor_init)
{
    fl::Gaussian<DVector> gaussian(6);
    EXPECT_NO_THROW(test_gaussian_covariance(gaussian));
}

TEST_F(GaussianTests, dynamic_gaussian_covariance_SetStandard_init)
{
    fl::Gaussian<DVector> gaussian;
    gaussian.Dimension(7);
    EXPECT_NO_THROW(test_gaussian_covariance(gaussian));
}

TEST_F(GaussianTests, dynamic_uninitialized_gaussian)
{
    fl::Gaussian<DVector> gaussian;
    EXPECT_THROW(gaussian.Covariance(), fl::GaussianUninitializedException);
    EXPECT_THROW(gaussian.Precision(), fl::GaussianUninitializedException);
    EXPECT_THROW(gaussian.SquareRoot(), fl::GaussianUninitializedException);

    gaussian.Dimension(1);
    EXPECT_NO_THROW(gaussian.Covariance());
    EXPECT_NO_THROW(gaussian.Precision());
    EXPECT_NO_THROW(gaussian.SquareRoot());

    gaussian.Dimension(0);
    EXPECT_THROW(gaussian.Covariance(), fl::GaussianUninitializedException);
    EXPECT_THROW(gaussian.Precision(), fl::GaussianUninitializedException);
    EXPECT_THROW(gaussian.SquareRoot(), fl::GaussianUninitializedException);
}

