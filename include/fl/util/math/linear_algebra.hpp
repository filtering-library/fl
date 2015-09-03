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
 * \file linear_algebra.hpp
 * \date January 2015
 * \author Jan Issac (jan.issac@gmail.com)
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#pragma once


#include <Eigen/Dense>

#include <cmath>
#include <vector>

namespace fl
{

/**
 * \ingroup linear_algebra
 *
 * \brief Blockweise matrix inversion using the Sherman-Morrision-Woodbury
 * indentity given that \f$\Sigma^{-1}_{aa}\f$ of
 *
 * \f$
 *  \begin{pmatrix} \Sigma_{aa} & \Sigma_{ab} \\
 *                  \Sigma_{ba} & \Sigma_{bb} \end{pmatrix}^{-1}
 * \f$
 *
 * is already available.
 *
 * \f$
 *  \begin{pmatrix} \Sigma_{aa} & \Sigma_{ab} \\
 *                  \Sigma_{ba} & \Sigma_{bb} \end{pmatrix}^{-1} =
 * \begin{pmatrix} \Lambda_{aa} & \Lambda_{ab} \\
 *                 \Lambda_{ba} & \Lambda_{bb} \end{pmatrix} = \Lambda
 * \f$
 *
 * \param [in]  A_inv   \f$ \Sigma^{-1}_{aa} \f$
 * \param [in]  B       \f$ \Sigma_{ab} \f$
 * \param [in]  C       \f$ \Sigma_{ba} \f$
 * \param [in]  D       \f$ \Sigma_{bb} \f$
 * \param [out] L_A     \f$ \Lambda_{aa} \f$
 * \param [out] L_B     \f$ \Lambda_{ab} \f$
 * \param [out] L_C     \f$ \Lambda_{ba} \f$
 * \param [out] L_D     \f$ \Lambda_{bb} \f$
 */
template <typename MatrixAInv,
          typename MatrixB,
          typename MatrixC,
          typename MatrixD,
          typename MatrixLA,
          typename MatrixLB,
          typename MatrixLC,
          typename MatrixLD>
inline void smw_inverse(const MatrixAInv& A_inv,
                  const MatrixB& B,
                  const MatrixC& C,
                  const MatrixD& D,
                  MatrixLA& L_A,
                  MatrixLB& L_B,
                  MatrixLC& L_C,
                  MatrixLD& L_D)
{
    auto&& AinvB = (A_inv * B).eval();

    L_D = (D - C * AinvB).inverse();
    L_C = -(L_D * C * A_inv).eval();
    L_A = A_inv - AinvB * L_C;
    L_B = L_C.transpose();
}

/**
 * \ingroup linear_algebra
 *
 * \brief Blockweise matrix inversion using the Sherman-Morrision-Woodbury
 * indentity given that \f$\Sigma^{-1}_{aa}\f$ of
 *
 * \f$
 *  \begin{pmatrix} \Sigma_{aa} & \Sigma_{ab} \\
 *                  \Sigma_{ba} & \Sigma_{bb} \end{pmatrix}^{-1}
 * \f$
 *
 * is already available.
 *
 * \f$
 *  \begin{pmatrix} \Sigma_{aa} & \Sigma_{ab} \\
 *                  \Sigma_{ba} & \Sigma_{bb} \end{pmatrix}^{-1} =
 * \begin{pmatrix} \Lambda_{aa} & \Lambda_{ab} \\
 *                 \Lambda_{ba} & \Lambda_{bb} \end{pmatrix} = \Lambda
 * \f$
 *
 * \param [in]  A_inv   \f$ \Sigma^{-1}_{aa} \f$
 * \param [in]  B       \f$ \Sigma_{ab} \f$
 * \param [in]  C       \f$ \Sigma_{ba} \f$
 * \param [in]  D       \f$ \Sigma_{bb} \f$
 * \param [out] L_A     \f$ \Lambda_{aa} \f$
 * \param [out] L_B     \f$ \Lambda_{ab} \f$
 * \param [out] L_C     \f$ \Lambda_{ba} \f$
 * \param [out] L_D     \f$ \Lambda_{bb} \f$
 * \param [out] L       \f$ \Lambda \f$
 */
template <typename MatrixAInv,
          typename MatrixB,
          typename MatrixC,
          typename MatrixD,
          typename MatrixLA,
          typename MatrixLB,
          typename MatrixLC,
          typename MatrixLD,
          typename ResultMatrix>
inline void smw_inverse(const MatrixAInv& A_inv,
                  const MatrixB& B,
                  const MatrixC& C,
                  const MatrixD& D,
                  MatrixLA& L_A,
                  MatrixLB& L_B,
                  MatrixLC& L_C,
                  MatrixLD& L_D,
                  ResultMatrix& L)
{
     smw_inverse(A_inv, B, C, D, L_A, L_B, L_C, L_D);

    L.resize(L_A.rows() + L_C.rows(), L_A.cols() + L_B.cols());

    L.block(0,          0,          L_A.rows(), L_A.cols()) = L_A;
    L.block(0,          L_A.cols(), L_B.rows(), L_B.cols()) = L_B;
    L.block(L_A.rows(), 0,          L_C.rows(), L_C.cols()) = L_C;
    L.block(L_A.rows(), L_A.cols(), L_D.rows(), L_D.cols()) = L_D;
}

/**
 * \ingroup linear_algebra
 *
 * \brief Blockweise matrix inversion using the Sherman-Morrision-Woodbury
 * indentity given that \f$\Sigma^{-1}_{aa}\f$ of
 *
 * \f$
 *  \begin{pmatrix} \Sigma_{aa} & \Sigma_{ab} \\
 *                  \Sigma_{ba} & \Sigma_{bb} \end{pmatrix}^{-1}
 * \f$
 *
 * is already available.
 *
 * \f$
 *  \begin{pmatrix} \Sigma_{aa} & \Sigma_{ab} \\
 *                  \Sigma_{ba} & \Sigma_{bb} \end{pmatrix}^{-1} =
 * \begin{pmatrix} \Lambda_{aa} & \Lambda_{ab} \\
 *                 \Lambda_{ba} & \Lambda_{bb} \end{pmatrix} = \Lambda
 * \f$
 *
 * \param [in]  A_inv   \f$ \Sigma^{-1}_{aa} \f$
 * \param [in]  B       \f$ \Sigma_{ab} \f$
 * \param [in]  C       \f$ \Sigma_{ba} \f$
 * \param [in]  D       \f$ \Sigma_{bb} \f$
 * \param [out] L       \f$ \Lambda \f$
 */
template <typename MatrixAInv,
          typename MatrixB,
          typename MatrixC,
          typename MatrixD,
          typename ResultMatrix>
inline void smw_inverse(const MatrixAInv& A_inv,
                  const MatrixB& B,
                  const MatrixC& C,
                  const MatrixD& D,
                  ResultMatrix& L)
{
    MatrixAInv L_A;
    MatrixB L_B;
    MatrixC L_C;
    MatrixD L_D;

    smw_inverse(A_inv, B, C, D, L_A, L_B, L_C, L_D, L);
}

/**
 * \ingroup linear_algebra
 *
 * \brief Normalizes the values of input vector such that their sum is equal to
 * the specified \c sum. For instance, any convex combination requires that the
 * weights of the weighted sum sums up to 1.
 */
template <typename T>
inline std::vector<T> normalize(const std::vector<T>& input, T sum)
{
    T old_sum = 0;
    for(size_t i = 0; i < input.size(); i++)
    {
        old_sum += input[i];
    }
    T factor = sum/old_sum;

    std::vector<T> output(input.size());
    for(size_t i = 0; i < input.size(); i++)
    {
        output[i] = factor*input[i];
    }

    return output;
}

/**
 * \ingroup linear_algebra
 * \brief Constructs the quaternion matrix for the specified quaternion vetcor
 *
 * \param q_xyzw  Quaternion vector
 *
 * \return Matrix representation of the quaternion vector
 */
inline Eigen::Matrix<double, 4, 3> quaternion_matrix(
        const Eigen::Matrix<double, 4, 1>& q_xyzw)
{
    Eigen::Matrix<double, 4, 3> Q;
    Q << q_xyzw(3),  q_xyzw(2), -q_xyzw(1),
        -q_xyzw(2),  q_xyzw(3),  q_xyzw(0),
         q_xyzw(1), -q_xyzw(0),  q_xyzw(3),
        -q_xyzw(0), -q_xyzw(1), -q_xyzw(2);

    return 0.5*Q;
}

/**
 * \ingroup linear_algebra
 * \todo DOC
 */
template <typename RegularMatrix, typename SquareRootMatrix>
void square_root(const RegularMatrix& regular_matrix,
                 SquareRootMatrix& square_root)
{
    square_root = regular_matrix.llt().matrixL();
}

/**
 * \ingroup linear_algebra
 * \todo DOC
 */
template <typename RegularMatrix>
void sqrt_diagonal(const RegularMatrix& regular_matrix,
                        RegularMatrix& square_root)
{
    square_root = regular_matrix;
    for (size_t i = 0; i < square_root.rows(); ++i)
    {
        square_root(i, i) = std::sqrt(square_root(i, i));
    }
}

/**
 * \ingroup linear_algebra
 * \brief Computes the square root of each element of the regular_matrix
 */
template <typename RegularMatrix, typename SquareRootVector>
void sqrt_diagonal_vector(const RegularMatrix& diagonal_vector,
                              SquareRootVector& square_root)
{
    square_root = diagonal_vector;
    for (size_t i = 0; i < square_root.rows(); ++i)
    {
        square_root(i, 0) = std::sqrt(square_root(i, 0));
    }
}

/**
 * \ingroup linear_algebra
 * \brief Inverts a diagonal matrix represented by a vector. The result is again
 *        a vector of the inverted diagonal elements.
 */
template <typename SrcDiagonalMatrix, typename DestDiagonalMatrix>
void invert_diagonal_vector(const SrcDiagonalMatrix& diagonal,
                            DestDiagonalMatrix& diagonal_inverse)
{
    diagonal_inverse = diagonal.array().cwiseInverse();

//    diagonal_inverse.resize(diagonal.size());
//    for (int i = 0; i < diagonal.size(); ++i)
//    {
//        diagonal_inverse(i) = 1./diagonal(i);
//    }
}

/**
 * \ingroup linear_algebra
 *
 * \brief Computes the Frobenius norm of a given real matrix.
 *
 * The Frobenius norm is given by
 *
 * \f$
 *   \| A \|_F := \sqrt{\sum\limits_{i=1}^m \sum\limits_{j=1}^n |a_{ij} |^2 }
 * \f$
 */
template <typename Derived>
double frobenius_norm(const Eigen::MatrixBase<Derived>& a)
{
    return std::sqrt(a.array().abs().square().sum());
}

/**
 * \ingroup linear_algebra
 * \brief Checks whether two matrices of the same size are similiar w.r.t. some
 * \f$\epsilon\f$.
 *
 * The similarity check is performed based on the Frobenius Norm
 * (\ref frobenius_norm):
 *
 * For any real matrix set \f$M \subseteq \mathbb{R}^{m\times n} \f$
 *
 * \f$ \text{are_similar}: (M \times M) \mapsto \{0, 1\}\f$
 *
 * \f$ \text{are_similar}(a, b)
 *     := \mathbf{1}_{[0, \epsilon)}\left(\| a - b \|_F\right)\f$
 * with \f$ a, b \in M\f$ and the indicator function
 *
 * \f$ \mathbf{1}_{[0, \epsilon)}(x) :=
 *     \begin{cases} 1 & x \in [0, \epsilon)
 *                  \\ 0, & \text{otherwise} \end{cases}\f$
 */
//template <typename DerivedA, typename DerivedB>
//bool are_similar(const Eigen::MatrixBase<DerivedA>& a,
//                 const Eigen::MatrixBase<DerivedB>& b,
//                 const double epsilon = 1.e-6)
//{
//    return frobenius_norm(a - b) < epsilon;
//}

template <typename DerivedA, typename DerivedB>
bool are_similar(const Eigen::MatrixBase<DerivedA>& a,
                 const Eigen::MatrixBase<DerivedB>& b,
                 const double epsilon = 1.e-6)
{
    //return (a - b).cwiseAbs().maxCoeff() < epsilon;
    return frobenius_norm(a - b) < epsilon;
}

/**
 * \ingroup linear_algebra
 * \brief Robust decomposition of M = L*L^T for positive semidefinite matrices
 */
template <typename Scalar, int Size>
Eigen::Matrix<Scalar, Size, Size>
matrix_sqrt(Eigen::Matrix<Scalar, Size, Size> M)
{
    typedef Eigen::Matrix<Scalar, Size, Size>   Matrix;
    typedef Eigen::Matrix<Scalar, Size, 1>      Vector;

    Eigen::LDLT<Matrix> ldlt;
    ldlt.compute(M);
    Vector D_sqrt = ldlt.vectorD();
    for(int i = 0; i < D_sqrt.rows(); ++i)
    {
        D_sqrt(i) = std::sqrt(std::fabs(D_sqrt(i)));
    }

    return ldlt.transpositionsP().transpose()
                    * (Matrix) ldlt.matrixL()
                    * D_sqrt.asDiagonal();
}

/**
 * \ingroup linear_algebra
 * \brief Checks wheather the specified dense matrix has a diagonal form
 *
 * \return true if diagonal, false otherwise
 */
template <typename Derived>
bool is_diagonal(const Eigen::MatrixBase<Derived>& m)
{
    Derived z = m;
    z -= z.diagonal().asDiagonal();

    return z.isZero();
}

/**
 * \ingroup linear_algebra
 * \brief Robust AX=B solver where X is a vector or a matrix
 */
template <typename MatrixA, typename VectorsB>
VectorsB solve(const Eigen::MatrixBase<MatrixA>& A,
               const VectorsB& B)
{
    // householderQr() was not providing always a solution
    //VectorsB x = A.householderQr().solve(B).eval();

    VectorsB x = A.colPivHouseholderQr().solve(B).eval();
    return x; //RVO
}

}


