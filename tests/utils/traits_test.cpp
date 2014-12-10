/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Max-Planck-Institute for Intelligent Systems,
 *                     University of Southern California
 *    Jan Issac (jan.issac@gmail.com)
 *    Manuel Wuthrich (manuel.wuthrich@gmail.com)
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

#define EIGEN_RUNTIME_NO_MALLOC
#include <Eigen/Dense>
#include <fast_filtering/utils/traits.hpp>

TEST(TraitsTests, is_dynamic)
{
    EXPECT_TRUE(fl::IsDynamic<Eigen::Dynamic>());
    EXPECT_FALSE(fl::IsDynamic<10>());
}

TEST(TraitsTests, is_fixed)
{
    EXPECT_FALSE(fl::IsFixed<Eigen::Dynamic>());
    EXPECT_TRUE(fl::IsFixed<10>());
}

TEST(TraitsTests, DimensionOf_dynamic)
{
    EXPECT_EQ(fl::DimensionOf<Eigen::MatrixXd>(), 0);
    EXPECT_EQ(fl::DimensionOf<Eigen::VectorXd>(), 0);

    typedef Eigen::Matrix<double, 10, -1> PartiallyDynamicMatrix;
    EXPECT_EQ(fl::DimensionOf<PartiallyDynamicMatrix>(), 0);
}

TEST(TraitsTests, DimensionOf_fixed)
{
    EXPECT_EQ(fl::DimensionOf<Eigen::Matrix3d>(), 3);
    EXPECT_EQ(fl::DimensionOf<Eigen::Vector4d>(), 4);

    typedef Eigen::Matrix<double, 10, 10> FixedMatrix;
    EXPECT_EQ(fl::DimensionOf<FixedMatrix>(), 10);

    typedef Eigen::Matrix<double, 10, 1> FixedVector;
    EXPECT_EQ(fl::DimensionOf<FixedVector>(), 10);
}



//TEST(EigenMemAllocTests, fixed)
//{
//    typedef Eigen::Matrix<double, 10, 1> Vector;
//    typedef Eigen::Matrix<double, 10, 10> Matrix;
//    Eigen::internal::set_is_malloc_allowed(false);
//    Vector v;
//    Vector r;
//    Matrix R = Matrix::Random();
//    R *= R;
//    v.setRandom();
//    r = R * v;
//    R = R.inverse();

//    Eigen::internal::set_is_malloc_allowed(true);
//}

//TEST(EigenMemAllocTests, dynamic)
//{
//    static constexpr size_t DIM = 100;
//    Eigen::internal::set_is_malloc_allowed(false);
//    typedef Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::AutoAlign, 10000, 1> Vector;
//    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign, 10000, 10000> Matrix;
//    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign, 10000, -1> MatrixD;
////    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
////    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
//    std::cout << "sizeof(Matrix) = " << sizeof(Matrix) << std::endl;
//    std::cout << "Matrix::MaxSizeAtCompileTime = " << Matrix::MaxSizeAtCompileTime << std::endl;
//    std::cout << "sizeof(MatrixDynamic) = " << sizeof(MatrixD) << std::endl;
//    std::cout << "MatrixDynamic::MaxSizeAtCompileTime = " << MatrixD::MaxSizeAtCompileTime << std::endl;
//    Matrix R = Matrix::Random(DIM, DIM);
//    Vector v(DIM);
//    Vector r(DIM);
//    R = R * R;
//    v.setRandom();
//    r.noalias() = R * v;
//    R.transposeInPlace();

//    Eigen::internal::set_is_malloc_allowed(true);
//}




//class MyObject
//{
//public:
//    MyObject() = default;

//    MyObject(std::string name)
//        : name_(name)
//    {
//        std::cout << "Created " << name << std::endl;
//    }

//    MyObject(MyObject&& other) noexcept
//        : name_(std::move(other.name_))
//    {
//        std::cout << "Moved " << name_ << std::endl;
//    }

//    MyObject(const MyObject& other)
//        : name_(other.name_)
//    {
//        std::cout << "Copied " << name_ << std::endl;
//    }

//    MyObject& operator= (const MyObject& other)
//    {
//        std::cout << "Copy assigned" << std::endl;

//        name_ = other.name_;
//        return *this ;
//    }

//    MyObject& operator= (MyObject&& other) noexcept
//    {
//        std::cout << "Move assigned" << std::endl;

//        name_ = std::move(other.name_);
//        return *this;
//    }

//    void name(std::string name) { name_ = name; }
//    std::string name() const { return name_; }

//private:
//    std::string name_;
//};

//MyObject foo(MyObject obj)
//{
//    std::cout << "renamed " <<  obj.name();
//    obj.name("foo-" + obj.name());
//    std::cout << " to " << obj.name() << std::endl;
//    return obj;
//}


//TEST(MoveSemantics, someTests)
//{
//    MyObject ret_obj;

//    MyObject obj1 = {"guinea pig 1"};
//    MyObject obj2 = {"guinea pig 2"};

//    std::cout << "calling foo(obj)..." << std::endl;
//    ret_obj = foo(obj1);
//    std::cout << "got " << ret_obj.name() << std::endl;

//    std::cout << std::endl << "calling foo(move(obj))..." << std::endl;
//    ret_obj = foo(std::move(obj2));
//    std::cout << "got " << ret_obj.name() << std::endl;

//    std::cout << std::endl << "calling foo({\"name\"})..." << std::endl;
//    ret_obj = foo({"nameless guinea pig"});
//    std::cout << "got " << ret_obj.name() << std::endl;

//    std::cout << std::endl << "calling foo(MyObject(\"name\"))..." << std::endl;
//    ret_obj = foo(MyObject("another nameless guinea pig"));
//    std::cout << "got " << ret_obj.name() << std::endl;
//}



//static constexpr size_t number_of_points(int dimension)
//{
//    return (dimension > Eigen::Dynamic) ? 2 * dimension + 1 : 0;
//}

//TEST(conststuff, constexpressions)
//{
//    enum
//    {
//        JointDimension = fl::JoinSizes<1, 2, 3, 4>::Size,
//        NumberOfPoints = number_of_points(fl::JoinSizes<1, 2, 3, 4>::Size)
//    };

//    std::cout << JointDimension << std::endl;
//    std::cout << NumberOfPoints << std::endl;
//}
















