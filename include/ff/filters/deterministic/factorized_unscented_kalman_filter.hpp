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
 * \date 2014
 * \author Jan Issac (jan.issac@gmail.com)
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * Max-Planck-Institute for Intelligent Systems, University of Southern California
 */

#ifndef FAST_FILTERING_FILTERS_DETERMINISTIC_FACTORIZED_UNSCENTED_KALMAN_FILTER_HPP
#define FAST_FILTERING_FILTERS_DETERMINISTIC_FACTORIZED_UNSCENTED_KALMAN_FILTER_HPP

#include <Eigen/Dense>

#include <cmath>
#include <type_traits>
#include <memory>

#include <fl/util/assertions.hpp>
#include <fl/distribution/interface/standard_gaussian_mapping.hpp>
#include <fl/distribution/sum_of_deltas.hpp>
#include <ff/filters/deterministic/composed_state_distribution.hpp>

#include <fl/util/profiling.hpp>

namespace fl
{

/**
 * \class FactorizedUnscentedKalmanFilter
 *
 * The Factorized UKF filters high dimensional states using high dimensional
 * measurements. The state is composed of a unfactorized low-dimensional part
 * \f$a_t\f$ and a high-dimensional fully factorised segment
 * \f$b^{[1]}_t, b^{[2]}_t, \ldots, b^{[M]}_t \f$. The two parts are predicted
 * using two different process models ProcessModelA and ProcessModelB.
 */
template<typename CohesiveStateProcessModel,
         typename FactorizedStateProcessModel,
         typename ObservationModel>
class FactorizedUnscentedKalmanFilter
{
public:
    typedef typename Traits<CohesiveStateProcessModel>::State State_a;
    typedef typename Traits<CohesiveStateProcessModel>::Noise Noise_a;
    typedef typename Traits<CohesiveStateProcessModel>::Input Input_a;

    typedef typename Traits<FactorizedStateProcessModel>::State State_b_i;
    typedef typename Traits<FactorizedStateProcessModel>::Noise Noise_b_i;
    typedef typename Traits<FactorizedStateProcessModel>::Input Input_b_i;
        
    typedef typename ObservationModel::Obsrv Obsrv;

    typedef ComposedStateDistribution<State_a, State_b_i, Obsrv> StateDistribution;
    typedef typename StateDistribution::JointPartitions JointPartitions;
    typedef typename StateDistribution::Cov_aa Cov_aa;
    typedef typename StateDistribution::Cov_bb Cov_bb;
    typedef typename StateDistribution::Cov_yy Cov_yy;
    typedef typename StateDistribution::Cov_ay Cov_ay;
    typedef typename StateDistribution::Cov_ab Cov_ab;
    typedef typename StateDistribution::Cov_by Cov_by;

    typedef Eigen::Matrix<typename StateDistribution::Scalar,
                          Eigen::Dynamic,
                          Eigen::Dynamic> SigmaPoints;

    typedef std::shared_ptr<CohesiveStateProcessModel> CohesiveStateProcessModelPtr;
    typedef std::shared_ptr<FactorizedStateProcessModel> FactorizedStateProcessModelPtr;
    typedef std::shared_ptr<ObservationModel> ObservationModelPtr;

    enum RandomVariableIndex { a = 0, Q_a , b_i, Q_b_i, R_y_i, y_i };

public:
    FactorizedUnscentedKalmanFilter(
            const CohesiveStateProcessModelPtr cohesive_state_process_model,
            const FactorizedStateProcessModelPtr factorized_state_process_model,
            const ObservationModelPtr observation_model,
            double kappa = 1.):
        f_a_(cohesive_state_process_model),
        f_b_(factorized_state_process_model),
        h_(observation_model),        
        kappa_(kappa)
    {
        static_assert(std::is_same<
                          typename Traits<CohesiveStateProcessModel>::Scalar,
                          typename Traits<FactorizedStateProcessModel>::Scalar
                      >::value,
                      "Scalar types of both models must be the same.");

//        static_assert_base(CohesiveStateProcessModel,
//                           StationaryProcessModel<State_a, Input_a>);
        static_assert_base(CohesiveStateProcessModel,
                           StandardGaussianMapping<State_a, Noise_a>);
//        static_assert_base(FactorizedStateProcessModel,
//                           StationaryProcessModel<State_b_i, Input_b_i>);
        static_assert_base(FactorizedStateProcessModel,
                           StandardGaussianMapping<State_b_i, Noise_b_i>);

        alpha_ = 1.6;
        beta_ = 2.;
        kappa_ = 0.;
    }

    virtual ~FactorizedUnscentedKalmanFilter() { }

    /**
     * Predicts the state for the next time step
     *
     * \param [in]  prior_state         State prior distribution
     * \param [out] predicted_state     Predicted state posterior distribution
     *
     * \note TESTED
     */
    void Predict(const StateDistribution& prior_state,
                 double delta_time,
                 StateDistribution& predicted_state)
    {
        const Eigen::MatrixXd noise_Q_a =
                Eigen::MatrixXd::Identity(Dim(Q_a), Dim(Q_a));

        const Eigen::MatrixXd noise_Q_bi =
                Eigen::MatrixXd::Identity(Dim(Q_b_i), Dim(Q_b_i));

        const Eigen::MatrixXd noise_R_yi =
                Eigen::MatrixXd::Identity(Dim(R_y_i), Dim(R_y_i));

        // compute the sigma point partitions X = [Xa  XQa  0(b^[i])  XQb  XR]
        // 0(b^[i]) is the place holder for the a b^[i]
        ComputeSigmaPointPartitions
        ({
            { prior_state.mean_a,                   prior_state.cov_aa },
            { Noise_a::Zero(Dim(Q_a),  1),          noise_Q_a },
            { State_b_i::Zero(Dim(b_i),  1),        Cov_bb(Dim(b_i),Dim(b_i)) },
            { Noise_b_i::Zero(Dim(Q_b_i), 1),       noise_Q_bi },
            { Eigen::MatrixXd::Zero(Dim(R_y_i), 1), noise_R_yi }
         },
         X_);

        // FOR ALL X_[a]
        f_a(X_[a], X_[Q_a], delta_time, X_[a]);

        mean(X_[a], predicted_state.mean_a);
        predicted_state.mean_a_predicted = predicted_state.mean_a;
        X_a_norm_ = X_[a];
        Normalize(predicted_state.mean_a, X_a_norm_);
        predicted_state.cov_aa = X_a_norm_ * X_a_norm_.transpose();

        predicted_state
                .joint_partitions
                .resize(prior_state.joint_partitions.size());

        // predict the joint state [a  b_i  y_i]
        for (size_t i = 0; i < prior_state.joint_partitions.size(); ++i)
        {
            ComputeSigmaPoints(prior_state.joint_partitions[i].mean_b,
                               prior_state.joint_partitions[i].cov_bb,
                               Dim(a) + Dim(Q_a),
                               X_[b_i]);

            f_b(X_[b_i], X_[Q_b_i], delta_time, X_[b_i]);
            h(X_[a], X_[b_i], X_[R_y_i], i, Y_);

            typename StateDistribution::JointPartitions& predicted_partition =
                    predicted_state.joint_partitions[i];

            mean(X_[b_i], predicted_partition.mean_b);
            mean(Y_, predicted_partition.mean_y);

            Normalize(predicted_partition.mean_b, X_[b_i]);
            Normalize(predicted_partition.mean_y, Y_);

            predicted_partition.cov_ab = X_a_norm_ * X_[b_i].transpose();
            predicted_partition.cov_ay = X_a_norm_ * Y_.transpose();
            predicted_partition.cov_bb = X_[b_i] * X_[b_i].transpose();

            predicted_partition.cov_by = X_[b_i] * Y_.transpose();
            predicted_partition.cov_yy = Y_ * Y_.transpose();
        }
    }

    /**
     * Update the predicted_state and store the result in the posterior_state
     * The update step involves updating the cohesive state followed by the
     * update of the factorized part
     *
     * \param [in]  predicted_state     Propagated state
     * \param [in]  y                   Measurement
     * \param [out] posterior_state     Updated posterior state
     *
     * \attention NEEDS TO BE TESTED
     */
    void Update(const StateDistribution& predicted_state,
                const Eigen::MatrixXd& y,
                StateDistribution& posterior_state)
    {
        Update_a(predicted_state, y, posterior_state);
        Update_b(predicted_state, y, posterior_state);
    }


    /**
     * Update the cohesive predicted_state part a assuming a multi-dimensoinal
     * measurements for each factor
     *
     * \param [in]  predicted_state     Propagated state
     * \param [in]  y                   Measurement
     * \param [out] posterior_state     Updated posterior state
     *
     * \attention NEEDS TO BE TESTED
     */
    template <typename RT = void>
    typename std::enable_if<Obsrv::SizeAtCompileTime != 1, RT>::type
    Update_a(const StateDistribution& predicted_state,
             const Eigen::MatrixXd& y,
             StateDistribution& posterior_state)
    {
        const size_t count_b = predicted_state.count_partitions();
        const size_t dim_y_i = Dim(y_i);

        posterior_state.cov_aa_inverse = predicted_state.cov_aa.inverse();
        const Cov_aa& cov_aa_inv = posterior_state.cov_aa_inverse;

        Cov_yy cov_yy_given_a_inv_i;
        Eigen::MatrixXd A_i(dim_y_i, Dim(a));
        Eigen::MatrixXd T_i(Dim(a), dim_y_i);
        Cov_aa C(Dim(a), Dim(a));
        Eigen::MatrixXd D(Dim(a), 1);

        C.setZero();
        D.setZero();

        for (size_t i = 0; i < count_b; ++i)
        {
            if (std::isnan(y(i, 0)))
                    //|| std::fabs(y(i, 0) - partition.mean_y(0,0)) > 0.08)
            {
                continue;
            }

            const JointPartitions& partition
                    = predicted_state.joint_partitions[i];

            const Cov_ay& cov_ay = partition.cov_ay;
            const Cov_yy& cov_yy = partition.cov_yy;

            A_i = cov_ay.transpose() * cov_aa_inv;
            cov_yy_given_a_inv_i
                    = (cov_yy - cov_ay.transpose() * cov_aa_inv * cov_ay)
                      .inverse();

            T_i = A_i.transpose() * cov_yy_given_a_inv_i;

            C += T_i * A_i;
            D += T_i * (y.middleRows(i * dim_y_i, dim_y_i) - partition.mean_y);
        }

        if (!D.isZero())
        {
            posterior_state.cov_aa
                    = (cov_aa_inv + C).inverse();
            posterior_state.mean_a
                    = predicted_state.mean_a + posterior_state.cov_aa * D;
        }
        else
        {
            std::cout << "No valid measurements. Cohesive state partition has"
                         " no been updated." << std::endl;
            return;
        }
    }

    /**
     * Update the cohesive predicted_state part a assuming a one-dimensoinal
     * measurements for each factor
     *
     * \param [in]  predicted_state     Propagated state
     * \param [in]  y                   Measurement
     * \param [out] posterior_state     Updated posterior state
     *
     * \attention NEEDS TO BE TESTED
     */
    template <typename RT = void>
    typename std::enable_if<Obsrv::SizeAtCompileTime == 1, RT>::type
    Update_a(const StateDistribution& predicted_state,
             const Eigen::MatrixXd& y,
             StateDistribution& posterior_state)
    {
        size_t dim_b = predicted_state.count_partitions();

        Eigen::MatrixXd A(dim_b, Dim(a));
        Eigen::MatrixXd mu_y(dim_b, 1);
        Eigen::MatrixXd cov_yy_given_a_inv(dim_b, 1);

        posterior_state.cov_aa_inverse = predicted_state.cov_aa.inverse();
        Cov_aa& cov_aa_inv = posterior_state.cov_aa_inverse;

        size_t k = 0;
        Eigen::MatrixXd valid_y(dim_b, 1);
        for (size_t i = 0; i < dim_b; ++i)
        {
            const JointPartitions& partition
                    = predicted_state.joint_partitions[i];

            if (std::isnan(y(i, 0))
                    || std::fabs(y(i, 0) - partition.mean_y(0,0)) > 0.08)
            {
                continue;
            }

            const Cov_ay& cov_ay = partition.cov_ay;
            const Cov_yy& cov_yy = partition.cov_yy;

            A.row(k) = cov_ay.transpose();
            mu_y.row(k) = partition.mean_y;

            cov_yy_given_a_inv.row(k) =
                    cov_yy - cov_ay.transpose() * cov_aa_inv * cov_ay;

            valid_y.row(k) = y.row(i);

            k++;
        }

        if (!k)
        {
            std::cout << "No valid measurements. Cohesive state partition has"
                         " no been updated." << std::endl;
            return;
        }

        A.conservativeResize(k, Eigen::NoChange);
        mu_y.conservativeResize(k, Eigen::NoChange);
        cov_yy_given_a_inv.conservativeResize(k, Eigen::NoChange);
        valid_y.conservativeResize(k, Eigen::NoChange);

        A = A * cov_aa_inv;

        invert_diagonal_Vector(cov_yy_given_a_inv, cov_yy_given_a_inv);
        Eigen::MatrixXd AT_Cov_yy_given_a =
                A.transpose() * cov_yy_given_a_inv.asDiagonal();

        // update cohesive state segment
        posterior_state.cov_aa = (cov_aa_inv + AT_Cov_yy_given_a * A).inverse();
        posterior_state.mean_a =
                predicted_state.mean_a +
                posterior_state.cov_aa * (AT_Cov_yy_given_a * (valid_y - mu_y));
    }

    /**
     * Update the factorized predicted_state part b given the updated cohesive
     * part a.
     *
     * \param [in]  predicted_state     Propagated state
     * \param [in]  y                   Measurement
     * \param [out] posterior_state     Updated posterior state
     *
     * \attention NEEDS TO BE TESTED
     */
    void Update_b(const StateDistribution& predicted_state,
                  const Eigen::MatrixXd& y,
                  StateDistribution& posterior_state)
    {
        Eigen::MatrixXd L;
        Eigen::MatrixXd L_aa;
        Eigen::MatrixXd L_ay;
        Eigen::MatrixXd L_ya;
        Eigen::MatrixXd L_yy;

        const size_t count_b = predicted_state.count_partitions();

        Eigen::MatrixXd B;
        Eigen::MatrixXd c;
        Eigen::MatrixXd cov_ba_by;
        Eigen::MatrixXd K;
        Eigen::MatrixXd innov;
        Eigen::MatrixXd cov_b_given_a_y;

        cov_ba_by.resize(Dim(b_i), Dim(a) + Dim(y_i));
        innov.resize(Dim(a) + Dim(y_i), 1);
        innov.topRows(Dim(a)) = -predicted_state.mean_a_predicted;

        const Cov_aa& cov_aa_inv = posterior_state.cov_aa_inverse;

        for (size_t i = 0; i < count_b; ++i)
        {
            const JointPartitions& partition
                    = predicted_state.joint_partitions[i];

            if (std::isnan(y(i, 0)) ||
                std::fabs(y(i, 0) - partition.mean_y(0,0)) > 0.20)
            {
                continue;
            }

            const Cov_ay& cov_ay = partition.cov_ay;
            const Cov_ab& cov_ab = partition.cov_ab;
            const Cov_by& cov_by = partition.cov_by;
            const Cov_bb& cov_bb = partition.cov_bb;
            const Cov_yy& cov_yy = partition.cov_yy;

            smw_inverse(cov_aa_inv, cov_ay, cov_ay.transpose(), cov_yy,
                         L_aa, L_ay, L_ya, L_yy,
                         L);

            B = cov_ab.transpose() * L_aa  +  cov_by * L_ya;

            cov_ba_by.leftCols(Dim(a)) = cov_ab.transpose();
            cov_ba_by.rightCols(Dim(y_i)) = cov_by;

            K = cov_ba_by * L;

            innov.bottomRows(Dim(y_i))
                    = y.middleRows(i * Dim(y_i), Dim(y_i))
                      - partition.mean_y;

            c = partition.mean_b + K * innov;

            cov_b_given_a_y = cov_bb - K * cov_ba_by.transpose();

            // update b_[i]
            posterior_state.joint_partitions[i].mean_b
                     = B * posterior_state.mean_a + c;
            posterior_state.joint_partitions[i].cov_bb
                     = cov_b_given_a_y
                       + B * posterior_state.cov_aa * B.transpose();
        }
    }


public:
    void f_a(const SigmaPoints& prior_X_a,
             const SigmaPoints& noise_X_a,
             const double delta_time,
             SigmaPoints& predicted_X_a)
    {
        Input_a zero_input = Input_a::Zero(f_a_->InputDimension(), 1);

        for (size_t i = 0; i < prior_X_a.cols(); ++i)
        {
            f_a_->condition(delta_time, prior_X_a.col(i), zero_input);
            predicted_X_a.col(i)
                    = f_a_->map_standard_normal(noise_X_a.col(i));
        }
    }

    void f_b(const SigmaPoints& prior_X_b_i,
             const SigmaPoints& noise_X_b_i,
             const double delta_time,
             SigmaPoints& predicted_X_b_i)
    {
        Input_b_i zero_input = Input_b_i::Zero(f_b_->standard_variate_dimension(), 1);

        for (size_t i = 0; i < prior_X_b_i.cols(); ++i)
        {            
            f_b_->condition(delta_time, prior_X_b_i.col(i), zero_input);
            predicted_X_b_i.col(i)
                    = f_b_->map_standard_normal(noise_X_b_i.col(i));
        }
    }

    void h(const SigmaPoints& prior_X_a,
           const SigmaPoints& prior_X_b_i,
           const SigmaPoints& noise_X_y_i,
           const size_t index,
           SigmaPoints& predicted_X_y_i)
    {
        predicted_X_y_i.resize(Dim(y_i), prior_X_a.cols());

        for (size_t i = 0; i < prior_X_a.cols(); ++i)
        {
            h_->condition(prior_X_a.col(i), prior_X_b_i.col(i), i, index);
            predicted_X_y_i.col(i)
                    = h_->map_standard_normal(noise_X_y_i.col(i));
        }
    }

    /**
     * \brief Dim Returns the dimension of the specified random variable ID
     *
     * \param var_id    ID of the requested random variable
     */
    size_t Dim(RandomVariableIndex var_id)
    {
        switch(var_id)
        {
        case a:     return f_a_->dimension();
        case Q_a:   return f_a_->standard_variate_dimension();
        case b_i:   return f_b_->dimension();
        case Q_b_i: return f_b_->standard_variate_dimension();
        case R_y_i: return h_->standard_variate_dimension();
        case y_i:   return h_->dimension();
        }
    }

    /**
     * Computes the weighted mean of the given sigma points
     *
     * \note TESTED
     */
    template <typename MeanVector>
    void mean(const SigmaPoints& sigma_points, MeanVector& mean)
    {
        double w_0;
        double w_i;
        ComputeWeights(sigma_points.cols(), w_0, w_i);

        mean = w_0 * sigma_points.col(0);        

        for (size_t i = 1; i < sigma_points.cols(); ++i)
        {
            mean += w_i * sigma_points.col(i);
        }
    }

    /**
     * Normalizes the given sigma point such that they represent zero mean
     * weighted points.
     * \param [in]  mean          Mean of the sigma points
     * \param [in]  w             Weights of the points used to determine the
     *                            covariance
     * \param [out] sigma_points  The sigma point collection
     *
     * \note TESTED
     */
    template <typename MeanVector>    
    void Normalize(const MeanVector& mean, SigmaPoints& sigma_points)
    {
        double w_0_sqrt;
        double w_i_sqrt;
        ComputeWeights(sigma_points.cols(), w_0_sqrt, w_i_sqrt);

        w_0_sqrt += (1 - alpha_ * alpha_ + beta_);

        w_0_sqrt = std::sqrt(w_0_sqrt);
        w_i_sqrt = std::sqrt(w_i_sqrt);

        sigma_points.col(0) = w_0_sqrt * (sigma_points.col(0) - mean);

        for (size_t i = 1; i < sigma_points.cols(); ++i)
        {
            sigma_points.col(i) = w_i_sqrt * (sigma_points.col(i) - mean);
        }
    }

    /**
     * Computes the Unscented Transform weights according to the specified
     * sigmapoints
     *
     * \param [in]  number_of_sigma_points
     * \param [out] w_0_sqrt    The weight for the first sigma point
     * \param [out] w_i_sqrt    The weight for the remaining sigma points
     *
     * \note TESTED
     */
    void ComputeWeights(size_t number_of_sigma_points,
                        double& w_0,
                        double& w_i)
    {
        size_t dimension = (number_of_sigma_points - 1) / 2;

        double lambda = alpha_ * alpha_ * (double(dimension) + kappa_)
                        - double(dimension);
        w_0 = lambda / (double(dimension) + lambda);
        w_i = 1. / (2. * (double(dimension) + lambda));
    }

    /**
     * Computes the sigma point partitions for all specified moment pairs
     *
     * \param moments_list              Moment pair of each partition.
     *                                  For a place holder, the first moment
     *                                  (the mean) must have the required number
     *                                  of rows and 0 columns.
     * \param sigma_point_partitions    sigma point partitions
     *
     * \note TESTED
     */
    void ComputeSigmaPointPartitions(
            const std::vector<std::pair<Eigen::MatrixXd,
                                        Eigen::MatrixXd>>& moment_pairs,
            std::vector<SigmaPoints>& X)
    {
        size_t dim = 0;
        for (auto& moments : moment_pairs) { dim += moments.first.rows(); }
        size_t number_of_points = 2 * dim + 1;

        X.resize(moment_pairs.size());

        size_t offset = 0;
        size_t i = 0;
        for (auto& moments : moment_pairs)
        {
            X[i].resize(moments.first.rows(), number_of_points);

            // check whether this is only a place holder
            if (!moments.second.isZero())
            {
                ComputeSigmaPoints(moments.first, moments.second, offset, X[i]);
            }

            offset += moments.first.rows();
            i++;
        }
    }

    /**
     * Computes the sigma point partition. Given a random variable X and its
     * partitions [Xa  Xb  Xc] this function computes the sigma points of
     * a single partition, say Xb. This is done such that the number of sigma
     * points in Xb is 2*dim([Xa  Xb  Xc])+1. In so doing, the sigma points of X
     * can be computed partition wise, and ultimately augmented together.
     *
     * \param [in]  mean          First moment
     * \param [in]  covariance    Second centered moment
     * \param [in]  offset        Offset dimension if this transform is a
     *                            partition of a larger one
     * \param [out] sigma_points  Selected sigma points
     *
     * \note TESTED
     */
    template <typename MeanVector, typename CovarianceMatrix>
    void ComputeSigmaPoints(const MeanVector& mean,
                            const CovarianceMatrix& covariance,
                            const size_t offset,
                            SigmaPoints& sigma_points)
    {
        // assert sigma_points.rows() == mean.rows()
        size_t joint_dimension = (sigma_points.cols() - 1) / 2;
        CovarianceMatrix covarianceSqr = covariance.llt().matrixL();

        covarianceSqr *=
                std::sqrt(
                    (double(joint_dimension)
                     + alpha_ * alpha_ * (double(joint_dimension) + kappa_)
                     - double(joint_dimension)));

        //sigma_points.setZero();
        sigma_points.col(0) = mean;

        MeanVector pointShift;
        for (size_t i = 1; i <= joint_dimension; ++i)
        {
            if (offset + 1 <= i && i < offset + 1 + covariance.rows())
            {
                pointShift = covarianceSqr.col(i - (offset + 1));

                sigma_points.col(i) = mean + pointShift;
                sigma_points.col(joint_dimension + i) = mean - pointShift;
            }
            else
            {
                sigma_points.col(i) = mean;
                sigma_points.col(joint_dimension + i) = mean;
            }
        }
    }


public:
    CohesiveStateProcessModelPtr f_a_;
    FactorizedStateProcessModelPtr f_b_;
    ObservationModelPtr h_;

    double kappa_;
    double beta_;
    double alpha_;

    // sigma points
    std::vector<SigmaPoints> X_;
    SigmaPoints Y_;
    SigmaPoints X_a_norm_;
};

}

#endif
