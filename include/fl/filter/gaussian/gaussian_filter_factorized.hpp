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
 * \file gaussian_filter_factorized.cpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__FILTER__GAUSSIAN__GAUSSIAN_FILTER_FACTORIZED_HPP
#define FL__FILTER__GAUSSIAN__GAUSSIAN_FILTER_FACTORIZED_HPP

#include <omp.h>

#include <fl/util/meta.hpp>
#include <fl/util/traits.hpp>
#include <fl/util/profiling.hpp>

#include <fl/exception/exception.hpp>
#include <fl/filter/filter_interface.hpp>
#include <fl/filter/gaussian/point_set.hpp>
#include <fl/filter/gaussian/feature_policy.hpp>

#include <fl/distribution/gaussian.hpp>
#include <fl/distribution/joint_distribution.hpp>

#include <fl/model/process/joint_process_model.hpp>
#include <fl/model/observation/joint_observation_model.hpp>

namespace fl
{

// Forward declarations
template <typename...> class GaussianFilter;

/**
 * \defgroup gaussian_filter_iid Factorized Gaussian Filter (IID)
 * \ingroup sigma_point_kalman_filters
 */


#ifdef TEMPLATE_ARGUMENTS
#  undef TEMPLATE_ARGUMENTS
#endif

#define TEMPLATE_ARGUMENTS \
    JointProcessModel< \
        StateProcessModel, \
        JointProcessModel<MultipleOf<LocalParamModel, Count>>>, \
    Adaptive< \
        JointObservationModel< \
            MultipleOf<LocalObsrvModel, Count> \
        > \
    >, \
    PointSetTransform, \
    FeaturePolicy<>,\
    Options<FactorizeParams>

/**
 * \ingroup gaussian_filter_iid
 * Traits of Factorized GaussianFilter for IID Parameters
 */
#ifndef GENERATING_DOCUMENTATION
template <
    typename StateProcessModel,
    typename LocalParamModel,
    typename LocalObsrvModel,
    int Count,
    typename PointSetTransform,
    template <typename ...> class FeaturePolicy
>
#endif
struct Traits<GaussianFilter<TEMPLATE_ARGUMENTS>>
{
    /** \cond internal ====================================================== */
    /**
     * \brief Represents the factorized model of a set of independent parameters
     * which shall be filtered jointly with the state.
     */
    typedef JointProcessModel<
                MultipleOf<LocalParamModel, Count>
            > JointParamProcessModel;

    /**
     * \brief Internal joint process model consisting of \c StateProcessModel
     * and the JointProcessModel of multiple ParamProcessModel.
     */
    typedef JointProcessModel<
                StateProcessModel,
                JointParamProcessModel
            > ProcessModel;

    /**
     * \brief Represents the joint sensor observation mode. The joint sensor
     * assumes that all sensor are of the same kind and have the same model.
     */
    typedef Adaptive<
                JointObservationModel<
                    MultipleOf<LocalObsrvModel, Count>
                >
            > ObservationModel;

    /**
     * \brief Marginal distribution of the state component
     */
    typedef Gaussian<typename Traits<StateProcessModel>::State> LocalStateDistr;

    /**
     * \brief Marginal distribution of \f$i\f$-th parameter component
     */
    typedef Gaussian<typename Traits<LocalParamModel>::State> LocalParamDistr;

    /**
     * \brief Marginal distribution of the parameter components. The marginal
     * ifself consist of multiple Gaussian marginals, one for each parameter.
     */
    typedef JointDistribution<
                MultipleOf<LocalParamDistr, Count>
            > JointParamDistr;
    /** \endcond ============================================================ */

    /*
     * Required concept (interface) types
     *
     * - Ptr
     * - State
     * - Input
     * - Observation
     * - Belief
     */
    //typedef std::shared_ptr<Filter> Ptr;
    typedef typename Traits<ProcessModel>::State State;
    typedef typename Traits<ProcessModel>::Input Input;
    typedef typename Traits<ObservationModel>::Obsrv Obsrv;

    /**
     * \brief This represents the joint state distribution.
     * The joint state distribution second centered moment is composed as
     * \f$\Sigma = \begin{bmatrix} \Sigma_a & 0 \\ 0 & \Sigma_b \end{bmatrix}\f$
     * where \f$\Sigma_a\f$ is the second moment of the marginal distribution of
     * the state component, and \f$\Sigma_b\f$ is the second moment of the
     * joint parameter moments
     * \f$ \Sigma_b = \begin{bmatrix}
     *                  \Sigma_{b_1} & 0 & 0 \\
     *                  0 & \ddots & 0 \\
     *                  0 & 0 & \Sigma_{b_n}
     *                \end{bmatrix}\f$
     */
    typedef JointDistribution<
                LocalStateDistr,
                JointParamDistr
            > Belief;

    /** \cond internal */
    /**
     * Feature type of an observaton.
     */
    typedef FeaturePolicy<Obsrv> FeatureMapping;
    typedef typename Traits<FeatureMapping>::ObsrvFeature ObsrvFeature;

    typedef typename Traits<ProcessModel>::Noise StateNoise;
    typedef typename Traits<ObservationModel>::Noise ObsrvNoise;

    typedef typename Traits<StateProcessModel>::State LocalState;
    typedef typename Traits<StateProcessModel>::Noise LocalStateNoise;
    typedef typename Traits<LocalParamModel>::State LocalParam;
    typedef typename Traits<LocalParamModel>::Noise LocalParamNoise;
    typedef typename Traits<LocalObsrvModel>::Obsrv LocalObsrv;
    typedef typename Traits<LocalObsrvModel>::Noise LocalObsrvNoise;

    typedef typename LocalState::Scalar Scalar;

    /**
     * \brief Represents the total number of points required by the point set
     * transform.
     */
    enum : signed int
    {
        NumberOfPoints = PointSetTransform::number_of_points(
                             JoinSizes<
                                 LocalState::RowsAtCompileTime,
                                 LocalStateNoise::RowsAtCompileTime,
                                 LocalParam::RowsAtCompileTime,
                                 LocalParamNoise::RowsAtCompileTime,
                                 LocalObsrvNoise::RowsAtCompileTime
                             >::Size)
    };

    typedef PointSet<State, NumberOfPoints> StatePointSet;
    typedef PointSet<Obsrv, NumberOfPoints> ObsrvPointSet;
    typedef PointSet<StateNoise, NumberOfPoints> StateNoisePointSet;
    typedef PointSet<ObsrvNoise, NumberOfPoints> ObsrvNoisePointSet;
    typedef PointSet<LocalState, NumberOfPoints> LocalStatePointSet;
    typedef PointSet<LocalParam, NumberOfPoints> LocalParamPointSet;
    typedef PointSet<LocalObsrv, NumberOfPoints> LocalObsrvPointSet;
    typedef PointSet<ObsrvFeature, NumberOfPoints> ObsrvFeaturePointSet;
    typedef PointSet<LocalStateNoise, NumberOfPoints> LocalStateNoisePointSet;
    typedef PointSet<LocalParamNoise, NumberOfPoints> LocalParamNoisePointSet;
    typedef PointSet<LocalObsrvNoise, NumberOfPoints> LocalObsrvNoisePointSet;

    typedef Eigen::Array<LocalParamPointSet, Count, 1> LocalParamPointSets;

    typedef Gaussian<LocalObsrv> LocalObsrvDistr;
    typedef Gaussian<LocalStateNoise> LocalStateNoiseDistr;
    typedef Gaussian<LocalParamNoise> LocalParamNoiseDistr;
    typedef Gaussian<LocalObsrvNoise> LocalObsrvNoiseDistr;
    /** \endcond */
};


/**
 * \ingroup gaussian_filter_iid
 *
 * This \c GaussianFilter represents a factorized implementation of a Sigma
 * Point Kalman Filter. The filter state consists of a coherent state component
 * and a factorized parameter component.
 *
 * \tparam JointProcessModel
 * \tparam JointObservationModel
 * \tparam PointSetTransform
 */
#ifndef GENERATING_DOCUMENTATION
template <
    typename StateProcessModel,
    typename LocalParamModel,
    typename LocalObsrvModel,
    int Count,
    typename PointSetTransform,
    template <typename...T> class FeaturePolicy
>
#endif
class GaussianFilter<
          JointProcessModel<
              StateProcessModel,
              JointProcessModel<MultipleOf<LocalParamModel, Count>>>,
          Adaptive<
              JointObservationModel<
                  MultipleOf<LocalObsrvModel, Count>
              >
          >,
          PointSetTransform,
          FeaturePolicy<>,
          Options<FactorizeParams>
      >
    :
    /* Implement the filter interface */
    public FilterInterface<GaussianFilter<TEMPLATE_ARGUMENTS>>
{
private:
    /** Typdef of \c This for #from_traits(TypeName) helper */
    typedef GaussianFilter This;

public:
    /* public concept interface types */
    typedef from_traits(State);
    typedef from_traits(Input);
    typedef from_traits(Obsrv);
    typedef from_traits(Belief);

    /* Model types */
    typedef from_traits(ProcessModel);
    typedef from_traits(ObservationModel);
    typedef from_traits(JointParamProcessModel);
    typedef from_traits(FeatureMapping);

private:
    /* Sigma Point Set types */
    typedef from_traits(StatePointSet);
    typedef from_traits(ObsrvPointSet);
    typedef from_traits(StateNoisePointSet);
    typedef from_traits(ObsrvNoisePointSet);
    typedef from_traits(LocalStatePointSet);
    typedef from_traits(LocalParamPointSets);
    typedef from_traits(ObsrvFeaturePointSet);

    /* Variate types */
    typedef from_traits(LocalState);
    typedef from_traits(LocalParam);
    typedef from_traits(LocalObsrv);
    typedef from_traits(ObsrvFeature);
    typedef from_traits(LocalStateNoise);
    typedef from_traits(LocalParamNoise);
    typedef from_traits(LocalObsrvNoise);

    /* Intermediate compile time dimensions */
    enum : signed int
    {
        CTDim_a     = LocalState::SizeAtCompileTime,
        CTDim_b_i   = LocalParam::SizeAtCompileTime,
        CTDim_y_i   = ExpandSizes<
                          LocalObsrv::SizeAtCompileTime,
                          FeatureMapping::feature_dimension(1)
                      >::Size,
        CTDim_b     = Traits<JointParamProcessModel>::State::SizeAtCompileTime,
        CTDim_a_y_i = JoinSizes<CTDim_a, CTDim_y_i>::Size
    };

    /* Intermediate types */
    typedef from_traits(Scalar);
    typedef from_traits(StateNoise);
    typedef from_traits(ObsrvNoise);

    typedef Eigen::Matrix<Scalar, CTDim_a,     1>           Ax1;
    typedef Eigen::Matrix<Scalar, CTDim_a,     CTDim_a>     AxA;
    typedef Eigen::Matrix<Scalar, CTDim_a,     CTDim_y_i>   AxY;
    typedef Eigen::Matrix<Scalar, CTDim_y_i,   CTDim_a>     YxA;
    typedef Eigen::Matrix<Scalar, CTDim_y_i,   CTDim_y_i>   YxY;
    typedef Eigen::Matrix<Scalar, CTDim_b_i,   CTDim_a_y_i> BxAY;
    typedef Eigen::Matrix<Scalar, CTDim_a_y_i, 1>           AYx1;
    typedef Eigen::Matrix<Scalar, CTDim_a_y_i, CTDim_a_y_i> AYxAY;
    typedef Eigen::Matrix<Scalar, CTDim_b_i,   CTDim_a>     BxA;
    typedef Eigen::Array<BxA,     Count,       1>           BAxN;

    /**
     * \enum Variate IDs
     */
    enum Variate
    {
        a = 0, /**< Coherent state vector component \f$a\f$ */
        b = 1, /**< Joint vector of factorized parameter component $ */
        v_a,   /**< Noise vector state component \f$a\f$ */
        v_b,   /**< Joint noise vector of factorized parameters */
        v,     /**< Joint State noise */
        b_i,   /**< Single parameter \f$b_i\f$ */
        v_b_i, /**< Noise vector of a singe parameter */
        y,     /**< Joint measurement */
        w,     /**< Joint measurement noise */
        y_i,   /**< Single measurement \f$y_i\f$ of \f$i\f$-th sensor */
        w_i,   /**< Noise vector of the \f$i\f$-th sensor */
        x,     /**< Dimension of the entire state [a b_1 b_2 ... b_n]*/
        u,
        u_a,
        u_b_i,
        z,
        z_i
    };
    /** \endcond  */

public:

    /**
     * Creates a factorized Gaussian filter
     *
     * \param state_process_model
     * \param parameter_process_model
     * \param obsrv_model
     * \param point_set_transform
     * \param parameter_count
     */
    GaussianFilter(
            const StateProcessModel& state_process_model,
            const LocalParamModel& param_process_model,
            const LocalObsrvModel& obsrv_model,
            const PointSetTransform& point_set_transform,
            const FeatureMapping& feature_mapping = FeatureMapping(),
            int param_count = Count)
        : f_(state_process_model, {param_process_model, param_count}),
          h_(Adaptive<ObservationModel>(obsrv_model, param_count)),
          transform_(point_set_transform),
          feature_mapping_(feature_mapping),
          param_count_(param_count),

          /*
           * Determine the marginal distribution dimension of
           * [a,  v_a,  b_i,  v_b_i,  w_i]
           */
          dim_marginal_(dim(a) + dim(v_a) + dim(b_i) + dim(v_b_i) + dim(w_i)),

          /*
           * Determine the number of points given the join marginal of
           *
           * [a,  v_a,  b_i,  v_b_i,  w_i]
           */
          point_count_(point_set_transform.number_of_points(dim_marginal_))
    {
        time_ = 0;

        assert(param_count > 0);

        X_.resize(dim(x), point_count_);
        Y_.resize(dim(y), point_count_);
        Y_f.resize(dim(z), point_count_);
        X_v_.resize(dim(v), point_count_);
        X_w_.resize(dim(w), point_count_);

        X_a_.resize(dim(a), point_count_);
        X_b_.resize(param_count, 1);
        for (int i = 0; i < param_count_; ++i)
        {
            X_b_(i).resize(dim(b_i), point_count_);
        }

        PV(point_count_);

        /* Create noise standard gaussian distributions */
//        auto N_a = Gaussian<LocalStateNoise>(dim(v_a));
//        auto N_b_i = Gaussian<LocalParamNoise>(dim(v_b_i));
//        auto N_y_i = Gaussian<LocalObsrvNoise>(dim(w_i));

//        /* Pre-Compute the sigma points of noise distributions */
//        typedef from_traits(LocalStateNoisePointSet);
//        typedef from_traits(LocalParamNoisePointSet);
//        typedef from_traits(LocalObsrvNoisePointSet);

//        auto X_v_a = LocalStateNoisePointSet(dim(v_a), point_count_);
//        auto X_v_b_i = LocalParamNoisePointSet(dim(v_b_i), point_count_);
//        auto X_w_y_i = LocalObsrvNoisePointSet(dim(w_i), point_count_);

//        int dim_offset = dim(a);
//        transform_.forward(N_a, dim_marginal_, dim_offset, X_v_a);

//        dim_offset += dim(v_a) + dim(b_i);
//        transform_.forward(N_b_i, dim_marginal_, dim_offset, X_v_b_i);

//        dim_offset += dim(v_b_i);
//        transform_.forward(N_y_i, dim_marginal_, dim_offset, X_w_y_i);

//        B.resize(param_count_, 1);
//        c.resize(param_count_, 1);
//        cov_b_given_ay.resize(param_count_, 1);

//        /* Create X_v and X_w joint sigma points */
//        X_v_.points().topRows(dim(v_a)) = X_v_a.points();
//        for (int i = 0; i < param_count; ++i)
//        {
//            X_v_.points().middleRows(dim(v_a) + i * dim(v_b_i), dim(v_b_i)) =
//                X_v_b_i.points();

//            X_w_.points().middleRows(i * dim(w_i), dim(w_i)) =
//                X_w_y_i.points();


//            B(i) = BxA::Zero(dim(b_i), dim(a));
//            c(i) = LocalParam::Zero(dim(b_i), 1);
//        }

        B.resize(param_count_, 1);
        c.resize(param_count_, 1);
        cov_b_given_ay.resize(param_count_, 1);

        /* Create noise standard gaussian distributions */
        auto N_a = Gaussian<LocalStateNoise>(dim(v_a));
        auto N_b_i = Gaussian<LocalParamNoise>(dim(v_b_i));
        auto N_y_i = Gaussian<LocalObsrvNoise>(dim(w_i));

        /* Pre-Compute the sigma points of noise distributions */
        typedef from_traits(LocalStateNoisePointSet);
        typedef from_traits(LocalParamNoisePointSet);
        typedef from_traits(LocalObsrvNoisePointSet);

        auto X_v_a = LocalStateNoisePointSet(dim(v_a), point_count_);
        auto X_v_b_i = LocalParamNoisePointSet(dim(v_b_i), point_count_);
        auto X_w_y_i = LocalObsrvNoisePointSet(dim(w_i), point_count_);

        int dim_offset = dim(a);
        transform_.forward(N_a, dim_marginal_, dim_offset, X_v_a);

        /* Create X_v and X_w joint sigma points */
        X_v_.points().topRows(dim(v_a)) = X_v_a.points();
        for (int i = 0; i < param_count_; ++i)
        {
            dim_offset = dim(a);

            dim_offset += dim(v_a) + dim(b_i);
            transform_.forward(N_b_i, dim_marginal_, dim_offset, X_v_b_i);

            dim_offset += dim(v_b_i);
            transform_.forward(N_y_i, dim_marginal_, dim_offset, X_w_y_i);

            X_v_.points().middleRows(dim(v_a) + i * dim(v_b_i), dim(v_b_i)) =
                X_v_b_i.points();

            X_w_.points().middleRows(i * dim(w_i), dim(w_i)) =
                X_w_y_i.points();

            B(i) = BxA::Zero(dim(b_i), dim(a));
            c(i) = LocalParam::Zero(dim(b_i), 1);
        }

        first_run = true;
        dump_values = false;
    }


    bool first_run;


    /**
     * \copydoc FilterInterface::predict
     */
    virtual void predict(double dt,
                         const Input& u,
                         const Belief& prior_belief,
                         Belief& predicted_belief)
    {
        if (first_run)
        {
            first_run = false;

            auto& distr_b = std::get<b>(prior_belief.distributions());
            for (int i = 0; i < param_count_; ++i)
            {
                cov_b_given_ay(i) = distr_b.distribution(i).covariance();
            }
        }

//        {

//            /* Create noise standard gaussian distributions */
//            auto N_a = Gaussian<LocalStateNoise>(dim(v_a));
//            auto N_b_i = Gaussian<LocalParamNoise>(dim(v_b_i));
//            auto N_y_i = Gaussian<LocalObsrvNoise>(dim(w_i));

//            /* Pre-Compute the sigma points of noise distributions */
//            typedef from_traits(LocalStateNoisePointSet);
//            typedef from_traits(LocalParamNoisePointSet);
//            typedef from_traits(LocalObsrvNoisePointSet);

//            auto X_v_a = LocalStateNoisePointSet(dim(v_a), point_count_);
//            auto X_v_b_i = LocalParamNoisePointSet(dim(v_b_i), point_count_);
//            auto X_w_y_i = LocalObsrvNoisePointSet(dim(w_i), point_count_);

//            int dim_offset = dim(a);
//            transform_.forward(N_a, dim_marginal_, dim_offset, X_v_a);

//            /* Create X_v and X_w joint sigma points */
//            X_v_.points().topRows(dim(v_a)) = X_v_a.points();
//            for (int i = 0; i < param_count_; ++i)
//            {                                                                                                                                              Been quite
//                dim_offset = dim(a);

//                dim_offset += dim(v_a) + dim(b_i);
//                transform_.forward(N_b_i, dim_marginal_, dim_offset, X_v_b_i);

//                dim_offset += dim(v_b_i);
//                transform_.forward(N_y_i, dim_marginal_, dim_offset, X_w_y_i);

//                X_v_.points().middleRows(dim(v_a) + i * dim(v_b_i), dim(v_b_i)) =
//                    X_v_b_i.points();

//                X_w_.points().middleRows(i * dim(w_i), dim(w_i)) =
//                    X_w_y_i.points();

//            }

//        }

        /*
         * Compute sigma points of part X_a and each of X_b(i)
         */
        transform(prior_belief, X_a_, X_b_);

        /*
         * X = [X_a^T  X_b^T]^T
         */
        augment(X_a_, X_b_, X_);

//        PV(X_.points());

        /*
         * Predict all sigma points X_[i]
         */
        for (int i = 0; i < point_count_; ++i)
        {
            X_[i] = f_.predict_state(dt, X_[i], X_v_[i], u);
        }

//        std::cout << "predicted X" << std::endl;
//        PV(X_.points());
    }

    /**
     * \copydoc FilterInterface::update
     */
    virtual void update(const Obsrv& obsrv,
                        const Belief& predicted_belief,
                        Belief& posterior_belief)
    {
        /* ================================================================== */
        /*                                                                    */
        /*                                                                    */
        /*                                                                    */
        /*                     ugglyness (╯°□°)╯︵ ┻━┻                        */
        /*                                                                    */
        /*                                                                    */
        /*                                                                    */
        /* ================================================================== */


        auto&& postr_a = std::get<a>(posterior_belief.distributions());
        auto&& postr_b = std::get<b>(posterior_belief.distributions())
                            .distributions();

        /* predict all observations */
        for (int i = 0; i < point_count_; ++i)
        {
            Y_[i] = h_.predict_obsrv(X_[i], X_w_[i], time_);
        }
        time_++;

        auto obsrv_prediction = Y_.mean();
        auto centered_prediction = Y_.centered_points();
        auto var = (centered_prediction.array().pow(2).rowwise().sum() / double(point_count_)).eval();

        for (int i = 0; i < point_count_; ++i)
        {
            Y_f[i] = feature_mapping_.extract(Y_[i], obsrv_prediction, var);
        }
        ObsrvFeature obsrv_feat
                = feature_mapping_.extract(obsrv, obsrv_prediction, var);

        // get dimension constants
        const int dim_a   = dim(a);
        const int dim_b_i = dim(b_i);
        const int dim_y_i = dim(z_i);

//        const auto mu_y = Y_f.center();
//        const auto mu_x = X_.center();
//        const auto mu_x_a = mu_x.topRows(dim(a)).eval();
//        const auto mu_x_b = mu_x.bottomRows(dim(b)).eval();
//        split(X_, X_a_, X_b_);

//        const auto W = X_a_.covariance_weights_vector();
//        const auto Y = Y_f.points();
//        const auto X_a = X_a_.points();
//        const auto cov_aa = (X_a * W.asDiagonal()  * X_a.transpose()).eval();
//        const auto cov_aa_inv = cov_aa.inverse().eval();

//        auto C = AxA::Zero(dim_a, dim_a).eval();
//        auto D = Ax1::Zero(dim_a, 1).eval();
//        auto B = BAxN(param_count_, 1);

//        auto innov_b_i = AYx1(dim_a + dim_y_i, 1);

        using namespace Eigen;

        MatrixXd mu_y = Y_f.center();
        MatrixXd mu_x = X_.center();
        MatrixXd mu_x_a = mu_x.topRows(dim(a)).eval();
        MatrixXd mu_x_b = mu_x.bottomRows(dim(b)).eval();
        split(X_, X_a_, X_b_);
        MatrixXd W = X_a_.covariance_weights_vector();
        MatrixXd Y = Y_f.points();
        MatrixXd X_a = X_a_.points();
        MatrixXd cov_aa = X_a * W.asDiagonal()  * X_a.transpose();
        MatrixXd cov_aa_inv = cov_aa.inverse();
        MatrixXd C = AxA::Zero(dim_a, dim_a);
        MatrixXd D = Ax1::Zero(dim_a, 1);

        MatrixXd innov_b_i = AYx1(dim_a + dim_y_i, 1);
        innov_b_i.topRows(dim_a) = -mu_x_a;

//        if (dump_values) PV(mu_y);
//        if (dump_values) PV(mu_x);
//        PV(mu_x);
//        PV(cov_aa);
//        if (dump_values) PV(cov_aa_inv);

//        for (int i = 0; i < param_count_; ++i)
//        {
//            /* == common ==================================================== */
//            auto Y_i = Y.middleRows(i * dim_y_i, dim_y_i);
//            auto cov_ay_i = (X_a * W.asDiagonal() * Y_i.transpose()).eval();
//            auto cov_yy_i = (Y_i * W.asDiagonal() * Y_i.transpose()).eval();
//            auto innov = (obsrv_feat.middleRows(i * dim_y_i, dim_y_i) -
//                          mu_y.middleRows(i * dim_y_i, dim_y_i)).eval();

//            /* == part a ==================================================== */
//            const auto A_i = (cov_ay_i.transpose() * cov_aa_inv).eval();

//            const auto cov_yy_i_inv_given_a =
//                    (cov_yy_i - cov_ay_i.transpose() * cov_aa_inv * cov_ay_i)
//                    .inverse()
//                    .eval();

//            const auto T = (A_i.transpose() * cov_yy_i_inv_given_a).eval();

//            C += (T * A_i).eval();
//            D += (T * innov).eval();

//            /* == part b ==================================================== */
//            auto X_b_i = X_b_(i).points();
//            auto cov_ab = (X_a   * W.asDiagonal() * X_b_i.transpose()).eval();
//            auto cov_bb = (X_b_i * W.asDiagonal() * X_b_i.transpose()).eval();
//            auto cov_by = (X_b_i * W.asDiagonal() * Y_i.transpose()).eval();

//            AxA L_aa; YxY L_yy; AxY L_ay; YxA L_ya; AYxAY L;
//            smw_inverse(cov_aa_inv, cov_ay_i, cov_ay_i.transpose(), cov_yy_i,
//                        L_aa,       L_ay,     L_ya,     L_yy, // [out] block
//                        L);                                   // [out] joint

//            B(i) = (cov_ab.transpose() * L_aa + cov_by * L_ya).eval();

//            auto cov_ba_by = BxAY(dim_b_i, dim_a + dim_y_i);
//            cov_ba_by.leftCols(dim_a) = cov_ab.transpose();
//            cov_ba_by.rightCols(dim_y_i) = cov_by;

//            auto K = (cov_ba_by * L).eval();
//            auto cov_b_given_ay = (cov_bb - K * cov_ba_by.transpose()).eval();

//            innov_b_i.bottomRows(dim_y_i) = innov;

//            postr_b(i).mean(
//                mu_x_b.middleRows(i * dim_b_i, dim_b_i) + K * innov_b_i);
//            postr_b(i).covariance(cov_b_given_ay);
//        }

        for (int i = 0; i < param_count_; ++i)
        {
            /* == common ==================================================== */
            MatrixXd Y_i = Y.middleRows(i * dim_y_i, dim_y_i);
            MatrixXd cov_ay_i = X_a * W.asDiagonal() * Y_i.transpose();
            MatrixXd cov_yy_i = Y_i * W.asDiagonal() * Y_i.transpose();
            MatrixXd innov = (obsrv_feat.middleRows(i * dim_y_i, dim_y_i) -
                              mu_y.middleRows(i * dim_y_i, dim_y_i));

            /* == part a ==================================================== */
            MatrixXd A_i = cov_ay_i.transpose() * cov_aa_inv;

            MatrixXd cov_yy_i_inv_given_a =
                    (cov_yy_i - cov_ay_i.transpose() * cov_aa_inv * cov_ay_i)
                    .inverse();

            MatrixXd T = A_i.transpose() * cov_yy_i_inv_given_a;

            C += (T * A_i);
            D += (T * innov);

//            if (dump_values) PV(A_i);
//            if (dump_values) PV(cov_yy_i_inv_given_a);
//            if (dump_values) PV(C);

            /* == part b ==================================================== */
            MatrixXd X_b_i = X_b_(i).points();
            MatrixXd cov_ab = (X_a   * W.asDiagonal() * X_b_i.transpose());
            MatrixXd cov_bb = (X_b_i * W.asDiagonal() * X_b_i.transpose());
            MatrixXd cov_by = (X_b_i * W.asDiagonal() * Y_i.transpose());

            AxA L_aa; YxY L_yy; AxY L_ay; YxA L_ya; AYxAY L;
            smw_inverse(cov_aa_inv, cov_ay_i, cov_ay_i.transpose(), cov_yy_i,
                        L_aa,       L_ay,     L_ya,     L_yy, // [out] block
                        L);                                   // [out] joint

            B(i) = (cov_ab.transpose() * L_aa + cov_by * L_ya);

            MatrixXd cov_ba_by = BxAY(dim_b_i, dim_a + dim_y_i);
            cov_ba_by.leftCols(dim_a) = cov_ab.transpose();
            cov_ba_by.rightCols(dim_y_i) = cov_by;

            MatrixXd K = (cov_ba_by * L);
            cov_b_given_ay(i) = (cov_bb - K * cov_ba_by.transpose());

            innov_b_i.bottomRows(dim_y_i) = innov;
            c(i) = mu_x_b.middleRows(i * dim_b_i, dim_b_i) + K * innov_b_i;
        }

        if (dump_values)
        {
            PV(fl::initial_seed);
            PV(fl::seed_inc);
            exit(-1);
        }

        MatrixXd new_cov_aa = (cov_aa_inv + C).inverse() ;
        if (not new_cov_aa.ldlt().isPositive())
        {
            std::cout << ">>>>>> postr_a.covariance() not positive()" << std::endl;

            dump_values = true;
            update(obsrv, predicted_belief, posterior_belief);
            return;
        }

        postr_a.covariance((cov_aa_inv + C).inverse());
        postr_a.mean(mu_x_a + postr_a.covariance() * D);

        auto postr_mu_a = postr_a.mean();
        auto postr_cov_aa = postr_a.covariance();
        for (int i = 0; i < param_count_; ++i)
        {
            postr_b(i).mean(
                 B(i) * postr_mu_a + c(i));
            postr_b(i).covariance(
                cov_b_given_ay(i) + B(i) * postr_cov_aa * B(i).transpose());
        }

        // TODO REMOVE
        predicted_obsrv = obsrv_prediction;

        auto sv = postr_a.covariance().jacobiSvd();

        std::cout << "condition number "
                  << sv.singularValues()(0)
                     /
                     sv.singularValues()(sv.singularValues().size() - 1)
                  << std::endl;

        std::cout << "inv condition number "
                  << sv.singularValues()(sv.singularValues().size() - 1)
                     /
                     sv.singularValues()(0)
                  << std::endl;
    }

    bool dump_values;

    /**
     * \copydoc FilterInterface::predict_and_update
     */
    virtual void predict_and_update(double delta_time,
                                    const Input& input,
                                    const Obsrv& observation,
                                    const Belief& prior_belief,
                                    Belief& posterior_belief)
    {
        predict(delta_time, input, prior_belief, posterior_belief);
        update(observation, posterior_belief, posterior_belief);
    }

    ProcessModel& process_model() { return f_; }
    ObservationModel& obsrv_model() { return h_; }
    PointSetTransform& point_set_transform() { return transform_; }
    StateProcessModel& local_process_model() { return std::get<0>(f_.models()); }
    JointParamProcessModel& joint_param_model() { return std::get<1>(f_.models()); }
    LocalParamModel& local_param_model() { return joint_param_model().local_process_model(); }
    LocalObsrvModel& local_obsrv_model() { return h_.local_obsrv_model(); }
    FeatureMapping& feature_mapping() { return feature_mapping_; }

    const ProcessModel& process_model() const { return f_; }
    const ObservationModel& obsrv_model() const { return h_; }
    const PointSetTransform& point_set_transform() const { return transform_; }
    const StateProcessModel& local_process_model() const { return std::get<0>(f_.models()); }
    const JointParamProcessModel& joint_param_model() const  { return std::get<1>(f_.models()); }
    const LocalParamModel& local_param_model() const { return joint_param_model().local_process_model(); }
    const LocalObsrvModel& local_obsrv_model() const { return h_.local_obsrv_model(); }
    const FeatureMapping& feature_mapping() const { return feature_mapping_; }

    Belief create_state_distribution() const
    {
        typedef from_traits(LocalStateDistr);
        typedef from_traits(LocalParamDistr);
        typedef from_traits(JointParamDistr);

        auto state_distr = Belief(
                               LocalStateDistr(dim(a)),
                               JointParamDistr(
                                   LocalParamDistr(dim(b_i)),
                                   param_count_)
                            );

        return state_distr;
    }

public:
    /* == Helpers =========================================================== */
    /** \cond internal */

    /**
     * \return Dimension of the specified variate
     *
     * \throws Exception("Unknown variate component.")
     */
    const int dim(Variate variate) const
    {
        switch (variate)
        {
        case a:     return local_process_model().state_dimension();
        case v_a:   return local_process_model().noise_dimension();
        case b:     return joint_param_model().state_dimension();
        case v_b:   return joint_param_model().noise_dimension();
        case b_i:   return local_param_model().state_dimension();
        case v_b_i: return local_param_model().noise_dimension();
        case v:     return f_.noise_dimension();
        case y:     return h_.obsrv_dimension();
        case w:     return h_.noise_dimension();
        case y_i:   return local_obsrv_model().obsrv_dimension();
        case w_i:   return local_obsrv_model().noise_dimension();
        case x:     return f_.state_dimension();
        case u:     return f_.input_dimension();
        case u_a:   return local_process_model().input_dimension();
        case u_b_i: return local_param_model().input_dimension();
        case z:     return feature_mapping_.feature_dimension(dim(y));
        case z_i:   return feature_mapping_.feature_dimension(dim(y_i));

        default:
            // throw!
            throw Exception("Unknown variate component.");
        }
    }

    /**
     * Compute sigma points of part X_a and each of X_b(i) for a given
     * state distribution \c distr
     */
    template <class Xa, class Xb>
    void transform(const Belief& distr, Xa& x_a, Xb& x_b)
    {
        // transform X_a part
        auto&& prior_a = std::get<a>(distr.distributions());
        transform_.forward(prior_a, dim_marginal_, 0, x_a);

//        // transform all X_b(i)
//        auto&& prior_b = std::get<b>(distr.distributions());
//        const int offset = dim(a) + dim(v_a);

//        for (int i = 0; i < param_count_; ++i)
//        {
//            transform_.forward(
//                prior_b.distribution(i), dim_marginal_, offset, x_b(i));
//        }



        // transform all X_b(i)
//        auto&& prior_b = std::get<b>(distr.distributions());
//        const int offset = dim(a) + dim(v_a);

        Gaussian<LocalParam> g_b_i(dim(b_i));

        for (int i = 0; i < param_count_; ++i)
        {
            g_b_i.covariance(cov_b_given_ay(i));

            x_b(i).resize(dim(b_i), x_a.count_points());
            for (int k = 0; k < x_a.count_points(); ++k)
            {
                g_b_i.mean(B(i) * x_a[k] + c(i));
                x_b(i)[k] = g_b_i.sample();
            }
        }
    }

    template <class Xa, class Xb, class X_>
    void augment(const Xa& x_a, const Xb& x_b, X_& x)
    {
        x.points().topRows(dim(a)) = x_a.points();

        for (int i = 0; i < param_count_; ++i)
        {
            x.points().middleRows(dim(a) + i * dim(b_i), dim(b_i)) =
                x_b(i).points();
        }
    }

    template <class X_, class Xa, class Xb>
    void split(const X_& x, Xa& x_a, Xb& x_b)
    {
        x_a.points(x.points().topRows(dim(a)));

        for (int i = 0; i < param_count_; ++i)
        {
            x_b(i).points(
                x.points().middleRows(dim(a) + i * dim(b_i), dim(b_i)));
        }
    }

    /** \endcond */

public:
    Obsrv predicted_obsrv;
    double thresh;

//private:
    /** \cond internal */
    /* Model */
    ProcessModel f_;
    ObservationModel h_;
    PointSetTransform transform_;
    FeatureMapping feature_mapping_;

    // TODO fix this workaround
    double time_;
    BAxN B;
    Eigen::Array<Eigen::MatrixXd, -1, 1> c;
    Eigen::Array<Eigen::MatrixXd, -1, 1> cov_b_given_ay;
    /** \endcond */

protected:
    /** \cond internal */
    /* Data */
    const int param_count_;
    const int dim_marginal_;
    const int point_count_;

    StatePointSet X_;
    ObsrvPointSet Y_;
    ObsrvFeaturePointSet Y_f;
    StateNoisePointSet X_v_;
    ObsrvNoisePointSet X_w_;
    LocalStatePointSet X_a_;
    LocalParamPointSets X_b_;
    /** \endcond */
};


#ifndef GENERATING_DOCUMENTATION
template <
    typename StateProcessModel,
    typename LocalParamModel,
    typename LocalObsrvModel,
    int Count,
    typename PointSetTransform,
    template <typename...T> class FeaturePolicy
>
#endif
struct Traits<
           GaussianFilter<
               StateProcessModel,
               Join<MultipleOf<Adaptive<LocalObsrvModel, LocalParamModel>, Count>>,
               PointSetTransform,
               FeaturePolicy<>,
               Options<FactorizeParams>
           >
       >
    : Traits<GaussianFilter<TEMPLATE_ARGUMENTS>>
{ };

/**
 * \ingroup gaussian_filter_iid
 *
 * Translates the filter specification of the form
 *
 *   GaussianFilter< Process, Join<Many<Adaptive<Sensor, ParamModel>>> >
 *
 * to the form taking a joint process and a joint observation model
 *
 *   GaussianFilter< JointProcess, JointObsrv >
 */
# ifndef GENERATING_DOCUMENTATION
template <
    typename StateProcessModel,
    typename LocalObsrvModel,
    typename LocalParamModel,
    int Count,
    typename PointSetTransform,
    template <typename ...> class FeaturePolicy
>
#endif
class GaussianFilter<
          StateProcessModel,
          Join<MultipleOf<Adaptive<LocalObsrvModel, LocalParamModel>, Count>>,
          PointSetTransform,
          FeaturePolicy<>,
          Options<FactorizeParams>
        >
    : public GaussianFilter<TEMPLATE_ARGUMENTS>
{
public:
    typedef GaussianFilter<TEMPLATE_ARGUMENTS> Base;

    GaussianFilter(
        const StateProcessModel& state_process_model,
        const LocalParamModel& param_process_model,
        const LocalObsrvModel& obsrv_model,
        const PointSetTransform& point_set_transform,
        const typename Traits<Base>::FeatureMapping& feature_mapping
            = typename Traits<Base>::FeatureMapping(),
        int parameter_count = Count)
            : Base(state_process_model,
                   param_process_model,
                   obsrv_model,
                   point_set_transform,
                   feature_mapping,
                   parameter_count)
    { }
};

#ifdef TEMPLATE_ARGUMENTS
    #undef TEMPLATE_ARGUMENTS
#endif

}

#endif
