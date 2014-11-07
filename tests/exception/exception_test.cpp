

/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Max-Planck-Institute for Intelligent Systems,
 *                     University of Southern California
 *    Jan Issac (jan.issac@gmail.com)
 *    Manuel Wuthrich (manuel.wuthrich@gmail.com)
 *
 *  All rights reserved.
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
 * @author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * Max-Planck-Institute for Intelligent Systems, University of Southern California
 */

#include <gtest/gtest.h>

#include <fast_filtering/filtering_library/exception/exception.hpp>
#include <fast_filtering/filtering_library/filter/gaussian/point_set_gaussian.hpp>

TEST(Exception, create)
{
    struct FirstWeight
    {
        double w;
        std::string name;
    };

    typedef fl::PointSetGaussian<Eigen::Matrix<double, 1, 1>, -1> SigmaPointGaussian;
    SigmaPointGaussian sigmas(1);

    try
    {
        sigmas.point(0, Eigen::Matrix<double, 1, 1>::Random(), {1.23, 1.24});
        sigmas.point(1, Eigen::Matrix<double, 1, 1>::Random(), {1.23, 1.24});
        sigmas.point(2, Eigen::Matrix<double, 1, 1>::Random(), {1.23, 1.24});

        std::cout << sigmas.point(0) << std::endl;
        std::cout << sigmas.point(1) << std::endl;
        std::cout << sigmas.point(2) << std::endl;

//        std::cout << sigmas.parameter<0>(0).w << std::endl;
//        std::cout << sigmas.parameter<0>(0).name << std::endl;
//        std::cout << sigmas.parameter<1>(0) << std::endl;
//        std::cout << sigmas.parameter<0>(1).w << std::endl;
//        std::cout << sigmas.parameter<0>(1).name << std::endl;
//        std::cout << sigmas.parameter<1>(1) << std::endl;
//        std::cout << sigmas.parameter<1>(10) << std::endl;
    }
    catch(fl::OutOfBoundsException& e)
    {
        std::cout << boost::diagnostic_information(e) << std::endl;
    }
    catch(fl::Exception& e)
    {
        std::cout << boost::diagnostic_information(e) << std::endl;
    }
}


//TEST(Exception, createFilter)
//{
//    typedef fl::SigmaPointKalmanFilter<double> FilterAlgorithm;

//    fl::FilterInterface<FilterAlgorithm>* f = new FilterAlgorithm();

//    FilterAlgorithm::StateDistribution dist;

//    f->predict(0.03, dist, dist);
//    f->update(dist, 0, dist);

//    std::cout << dist << std::endl;

//    delete f;
//}
