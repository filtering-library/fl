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
 * \file decorrelated_gaussian.hpp
 * \date JUly 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once


#include <Eigen/Dense>

#include <vector>
#include <string>
#include <cstddef>
#include <type_traits>

#include <fl/util/traits.hpp>
#include <fl/exception/exception.hpp>
#include <fl/distribution/interface/moments.hpp>
#include <fl/distribution/interface/evaluation.hpp>
#include <fl/distribution/interface/standard_gaussian_mapping.hpp>
#include <fl/distribution/gaussian.hpp>

namespace fl
{

/**
 * \class DecorrelatedGaussian
 *
 * \brief General Decorrelated Gaussian Distribution
 * \ingroup distributions
 * \{
 *
 * The Gaussian is a general purpose distribution representing a multi-variate
 * \f${\cal N}(x; \mu, \Sigma)\f$. It can be used in various
 * ways while maintaining efficienty at the same time. This is due to it's
 * multi-representation structure. The distribution can be represented either by
 *
 *  - the covariance matrix \f$\Sigma\f$,
 *  - the precision matrix \f$\Sigma^{-1} = \Lambda\f$,
 *  - the covariance square root matrix (Cholesky decomposition or LDLT)
 *    \f$\sqrt{\Sigma} = L\sqrt{D}\f$,
 *  - or the diagonal form of the previous three options
 *    \f$diag(\sigma_1, \ldots, \sigma_n)\f$.
 *
 * A change in one representation results in change of all other
 * representations.
 *
 * Two key features of the distribution are its aibility to evaluation the
 * probability of a given sample and to map a noise sample into the distribution
 * sample space.
 *
 * \cond internal
 * The Gaussian internal structure uses lazy assignments or write on read
 * technique. Due to the multi-representation of the Gaussian, modifying one
 * representation affects all remaining ones. If one of the representation is
 * modified, the other representations are only then updated when needed. This
 * minimizes redundant computation and increases efficienty.
 * \endcond
 */
template <typename Variate>
class DecorrelatedGaussian
    : public Moments<Variate, typename DiagonalSecondMomentOf<Variate>::Type>,
      public Evaluation<Variate>,
      public StandardGaussianMapping<Variate, SizeOf<Variate>::Value>
{
public:
    typedef Evaluation<Variate> EvaluationInterface;

    typedef Moments<
                Variate, typename DiagonalSecondMomentOf<Variate>::Type
            > MomentsInterface;

    typedef StandardGaussianMapping<
                Variate, SizeOf<Variate>::Value
            > StdGaussianMappingInterface;

    /**
     * \brief Second moment matrix type, i.e covariance matrix, precision
     *        matrix, and their diagonal and square root representations
     */
    typedef typename DiagonalSecondMomentOf<Variate>::Type DiagonalSecondMoment;

    typedef typename SecondMomentOf<Variate>::Type DenseSecondMoment;

    /**
     * \brief Represents the StandardGaussianMapping standard variate type which
     *        is of the same dimension as the Gaussian Variate. The
     *        StandardVariate type is used to sample from a standard normal
     *        Gaussian and map it to this Gaussian
     */
    typedef
    typename StdGaussianMappingInterface::StandardVariate StandardVariate;

protected:
    /** \cond internal */
    /**
     * \enum Attribute
     * Implementation attributes. The enumeration lists the different
     * representations along with other properties such as the rank of the
     * second moment and the log normalizer.
     */
    enum Attribute
    {
        DiagonalCovarianceMatrix = 0,/**< Diagonal form of of cov. mat. */
        DiagonalPrecisionMatrix,     /**< Diagonal form of inv cov. mat. */
        DiagonalSquareRootMatrix,    /**< Diagonal form of Cholesky decomp. */
        Rank,                        /**< Covariance Rank */
        Normalizer,                  /**< Log probability normalizer */
        Determinant,                 /**< Determinant of covariance */

        Attributes                   /**< Total number of attribute */
    };
    /** \endcond */

public:
    /**
     * Creates a dynamic or fixed size Gaussian.
     *
     * \param dimension Dimension of the Gaussian. The default is defined by the
     *                  dimension of the variable type \em Vector. If the size
     *                  of the Vector at compile time is fixed, this will be
     *                  adapted. For dynamic-sized Variable the dimension is
     *                  initialized to 0.
     */
    explicit DecorrelatedGaussian(int dim = DimensionOf<Variate>()):
        StdGaussianMappingInterface(dim),
        dirty_(Attributes, true)
    {
        static_assert(SizeOf<Variate>::Value != 0, "Illegal static dimension");

        set_standard();
    }

    /**
     * \brief Overridable default destructor
     */
    virtual ~DecorrelatedGaussian() { }

    /**
     * \return Gaussian dimension
     */
    virtual constexpr int dimension() const
    {
        return StdGaussianMappingInterface::standard_variate_dimension();
    }

    /**
     * \return Gaussian first moment
     */
    virtual const Variate& mean() const
    {
        return mean_;
    }

    /**
     * \return Gaussian second centered moment
     *
     * Computes the covariance from other representation of not available
     *
     * \throws GaussianUninitializedException if the Gaussian is of dynamic-size
     *         and has not been initialized using SetStandard(dimension).
     * \throws InvalidGaussianRepresentationException if non-of the
     *         representation can be used as a source
     *
     * \cond internal
     * \pre |{Valid Representations}| > 0
     * \post {Valid Representations}
     *       = {Valid Representations} \f$ \cup \f$ {#CovarianceMatrix}
     * \endcond
     */
    virtual const DiagonalSecondMoment& covariance() const
    {
        if (dimension() == 0)
        {
            fl_throw(GaussianUninitializedException());
        }

        if (is_dirty(DiagonalCovarianceMatrix))
        {
            switch (select_first_representation({DiagonalSquareRootMatrix,
                                                 DiagonalPrecisionMatrix}))
            {
            case DiagonalSquareRootMatrix:
            {
                covariance_.diagonal() = square_root_.diagonal().cwiseProduct(
                                            square_root_.diagonal());
             } break;

            case DiagonalPrecisionMatrix:
            {
                covariance_.diagonal() = precision_.diagonal().cwiseInverse();
            } break;

            default:
                fl_throw(InvalidGaussianRepresentationException());
                break;
            }

            updated_internally(DiagonalCovarianceMatrix);
        }

        return covariance_;
    }

    /**
     * \return Gaussian second centered moment in the precision form (inverse
     * of the covariance)
     *
     * Computes the precision from other representation of not available
     *
     * \throws GaussianUninitializedException if the Gaussian is of dynamic-size
     *         and has not been initialized using SetStandard(dimension).
     * \throws InvalidGaussianRepresentationException if non-of the
     *         representation can be used as a source
     *
     * \cond internal
     * \pre |{Valid Representations}| > 0
     * \post {Valid Representations}
     *       = {Valid Representations} \f$ \cup \f$ {#PrecisionMatrix}
     * \endcond
     */
    virtual const DiagonalSecondMoment& precision() const
    {
        if (dimension() == 0)
        {
            fl_throw(GaussianUninitializedException());
        }

        if (is_dirty(DiagonalPrecisionMatrix))
        {
            switch (select_first_representation({DiagonalCovarianceMatrix,
                                                 DiagonalSquareRootMatrix}))
            {
            case DiagonalCovarianceMatrix:
            case DiagonalSquareRootMatrix:
                precision_.diagonal() = covariance().diagonal().cwiseInverse();
                break;

            default:
                fl_throw(InvalidGaussianRepresentationException());
                break;
            }

            updated_internally(DiagonalPrecisionMatrix);
        }

        return precision_;
    }


    /**
     * \return Gaussian second centered moment in the square root form (
     * Cholesky decomposition)
     *
     * Computes the square root from other representation of not available
     *
     * \throws GaussianUninitializedException if the Gaussian is of dynamic-size
     *         and has not been initialized using SetStandard(dimension).
     * \throws InvalidGaussianRepresentationException if non-of the
     *         representation can be used as a source
     *
     * \cond internal
     * \pre |{Valid Representations}| > 0
     * \post {Valid Representations}
     *       = {Valid Representations} \f$ \cup \f$ {#SquareRootMatrix}
     * \endcond
     */
    virtual const DiagonalSecondMoment& square_root() const
    {
        if (dimension() == 0)
        {
            fl_throw(GaussianUninitializedException());
        }

        if (is_dirty(DiagonalSquareRootMatrix))
        {
            switch (select_first_representation({DiagonalCovarianceMatrix,
                                                 DiagonalPrecisionMatrix}))
            {
            case DiagonalCovarianceMatrix:
            case DiagonalPrecisionMatrix:
            {
                square_root_.diagonal() = covariance().diagonal().cwiseSqrt();
            } break;

            default:
                fl_throw(InvalidGaussianRepresentationException());
                break;
            }

            updated_internally(DiagonalSquareRootMatrix);
        }

        return square_root_;
    }

    /**
     * \return True if the covariance matrix has a full rank
     *
     * \throws see covariance()
     *
     * \cond internal
     * \pre |{Valid Representations}| > 0
     * \post {Valid Representations}
     *       = {Valid Representations} \f$ \cup \f$ {#CovarianceMatrix}
     * \endcond
     */
    virtual bool has_full_rank() const
    {
        if (is_dirty(Rank))
        {
            full_rank_ = true;

            switch (select_first_representation({DiagonalCovarianceMatrix,
                                                 DiagonalPrecisionMatrix,
                                                 DiagonalSquareRootMatrix}))
            {
            case DiagonalCovarianceMatrix:
                full_rank_ = has_full_rank(covariance());
                break;
            case DiagonalPrecisionMatrix:
                full_rank_ = has_full_rank(precision());
                break;
            case DiagonalSquareRootMatrix:
                full_rank_ = has_full_rank(square_root());
                break;
            default:
                fl_throw(InvalidGaussianRepresentationException());
                break;
            }

            updated_internally(Rank);
        }

        return full_rank_;
    }

    /**
     * \return Log normalizing constant
     *
     * \throws see has_full_rank()
     *
     * \cond internal
     * \pre |{Valid Representations}| > 0
     * \post {Valid Representations}
     *       = {Valid Representations} \f$ \cup \f$ {#CovarianceMatrix}
     * \endcond
     */
    virtual Real log_normalizer() const
    {
        if (is_dirty(Normalizer))
        {
            if (has_full_rank())
            {
                log_norm_ = -0.5
                    * (log(covariance_determinant())
                       + Real(covariance().rows()) * log(2.0 * M_PI));
            }
            else
            {
                log_norm_ = 0.0; // FIXME
            }

            updated_internally(Normalizer);
        }

        return log_norm_;
    }

    /**
     * \return Covariance determinant
     *
     * \throws see covariance
     */
    virtual Real covariance_determinant() const
    {
        if (is_dirty(Determinant))
        {
            determinant_ = covariance().diagonal().prod();

            updated_internally(Determinant);
        }

        return determinant_;
    }

    /**
     * \return Log of the probability of the given sample \c vector
     *
     * \param vector sample which should be evaluated
     *
     * \throws see has_full_rank()
     *
     * \cond internal
     * \pre |{Valid Representations}| > 0
     * \post {Valid Representations}
     *       = {Valid Representations} \f$ \cup \f$ {#PrecisionMatrix}
     * \endcond
     */
    virtual Real log_probability(const Variate& vector) const
    {
        if(has_full_rank())
        {
            return log_normalizer() - 0.5
                    * (vector - mean()).transpose()
                    * precision()
                    * (vector - mean());
        }

        return -std::numeric_limits<Real>::infinity();
    }

    /**
     * \return a Gaussian sample of the type \c Vector determined by mapping a
     * noise sample into the Gaussian sample space
     *
     * \param sample    Noise Sample
     *
     * \throws see square_root()
     *
     * \cond internal
     * \pre |{Valid Representations}| > 0
     * \post {Valid Representations}
     *       = {Valid Representations} \f$ \cup \f$ {#SquareRootMatrix}
     * \endcond
     */
    virtual Variate map_standard_normal(const StandardVariate& sample) const
    {
        return mean() + square_root() * sample;
    }

    /**
     * Sets the Gaussian to a standard distribution with zero mean and identity
     * covariance.
     *
     * \cond internal
     * \pre {}
     * \post
     *  - Fully ranked covariance
     *  - {Valid representations} = {#CovarianceMatrix}
     * \endcond
     */
    virtual void set_standard()
    {
        mean_.resize(dimension());
        covariance_.resize(dimension());
        precision_.resize(dimension());
        square_root_.resize(dimension());

        mean(Variate::Zero(dimension()));

        auto cov = DiagonalSecondMoment(dimension());
        cov.setIdentity(dimension());
        covariance(cov);

        full_rank_ = true;
        updated_internally(Rank);
    }

    /**
     * Changes the dimension of the dynamic-size Gaussian and sets it to a
     * standard distribution with zero mean and identity covariance.
     *
     * \param new_dimension New dimension of the Gaussian
     *
     * \cond internal
     * \pre {}
     * \post
     *  - Fully ranked covariance
     *  - {Valid representations} = {#CovarianceMatrix}
     * \endcond
     *
     * \throws ResizingFixedSizeEntityException
     *         see GaussianMap::standard_variate_dimension(int)
     */
    virtual void dimension(int new_dimension)
    {
        StdGaussianMappingInterface::standard_variate_dimension(new_dimension);
        set_standard();
    }

    /**
     * Sets the mean
     *
     * \param mean New Gaussian mean
     *
     * \throws WrongSizeException
     */
    virtual void mean(const Variate& mean) noexcept
    {
        if (mean_.size() != mean.size())
        {
            fl_throw(fl::WrongSizeException(mean.size(), mean_.size()));
        }

        mean_ = mean;
    }

    /**
     * Sets the covariance matrix as a diagonal matrix
     *
     * \param diag_covariance New diagonal covariance matrix
     *
     * \cond internal
     * \pre |{Valid Representations}| > 0
     * \post {Valid Representations}
     *       = {Valid Representations} \f$ \cup \f$ {#DiagonalCovarianceMatrix}
     * \endcond
     *
     * \throws WrongSizeException
     */
    virtual void covariance(
        const DiagonalSecondMoment& diag_covariance) noexcept
    {
        if (diag_covariance.size() != covariance_.size())
        {
            fl_throw(
                fl::WrongSizeException(
                    diag_covariance.size(), covariance_.size()));
        }

        covariance_ = diag_covariance;
        updated_externally(DiagonalCovarianceMatrix);
    }

    /**
     * Sets the covariance matrix in its diagonal square root form
     *
     * \param diag_square_root New diagonal square root of the covariance
     *
     * \cond internal
     * \pre |{Valid Representations}| > 0
     * \post {Valid Representations}
     *       = {Valid Representations} \f$ \cup \f$ {#DiagonalSquareRootMatrix}
     * \endcond
     *
     * \throws WrongSizeException
     */
    virtual void square_root(
        const DiagonalSecondMoment& diag_square_root) noexcept
    {
        if (diag_square_root.size() != square_root_.size())
        {
            fl_throw(
                fl::WrongSizeException(
                    diag_square_root.size(), square_root_.size()));
        }

        square_root_ = diag_square_root;
        updated_externally(DiagonalSquareRootMatrix);
    }

    /**
     * Sets the covariance matrix in its diagonal precision form
     *
     * \param diag_precision New diagonal precision matrix
     *
     * \cond internal
     * \pre |{Valid Representations}| > 0
     * \post {Valid Representations}
     *       = {Valid Representations} \f$ \cup \f$ {#DiagonalPrecisionMatrix}
     * \endcond
     *
     * \throws WrongSizeException
     */
    virtual void precision(
        const DiagonalSecondMoment& diag_precision) noexcept
    {
        if (diag_precision.size() != precision_.size())
        {
            fl_throw(
                fl::WrongSizeException(
                    diag_precision.size(), precision_.size()));
        }

        precision_ = diag_precision;
        updated_externally(DiagonalPrecisionMatrix);
    }

    /**
     * Sets the covariance matrix as a diagonal matrix
     *
     * \param diag_covariance New diagonal covariance matrix
     *
     * \cond internal
     * \pre |{Valid Representations}| > 0
     * \post {Valid Representations}
     *       = {Valid Representations} \f$ \cup \f$ {#DiagonalCovarianceMatrix}
     * \endcond
     *
     * \throws WrongSizeException
     */
    virtual void covariance(
        const Eigen::MatrixBase<DenseSecondMoment>& cov) noexcept
    {
        covariance(cov.diagonal().asDiagonal());
    }

    /**
     * Sets the covariance matrix in its diagonal square root form
     *
     * \param diag_square_root New diagonal square root of the covariance
     *
     * \cond internal
     * \pre |{Valid Representations}| > 0
     * \post {Valid Representations}
     *       = {Valid Representations} \f$ \cup \f$ {#DiagonalSquareRootMatrix}
     * \endcond
     *
     * \throws WrongSizeException
     */
    virtual void square_root(
        const Eigen::MatrixBase<DenseSecondMoment>& sqrt) noexcept
    {
        square_root(sqrt.diagonal().asDiagonal());
    }

    /**
     * Sets the covariance matrix in its diagonal precision form
     *
     * \param diag_precision New diagonal precision matrix
     *
     * \cond internal
     * \pre |{Valid Representations}| > 0
     * \post {Valid Representations}
     *       = {Valid Representations} \f$ \cup \f$ {#DiagonalPrecisionMatrix}
     * \endcond
     *
     * \throws WrongSizeException
     */
    virtual void precision(
        const Eigen::MatrixBase<DenseSecondMoment>& prec) noexcept
    {
        precision(prec.diagonal().asDiagonal());
    }

protected:
    /** \cond internal */
    /**
     * Flags the specified attribute as valid and the rest of attributes as
     * dirty.
     *
     * \param attribute Modified attribute
     */
    virtual void updated_externally(Attribute attribute) const noexcept
    {
        std::fill(dirty_.begin(), dirty_.end(), true);
        updated_internally(attribute);
    }

    /**
     * Flags the specified attribute as valid.
     *
     * \param attribute Modified attribute
     */
    virtual void updated_internally(Attribute attribute) const noexcept
    {
        dirty_[attribute] = false;
    }

    /**
     * \return True if any of the other representation was modified.
     * \param attribute     Attribute in question
     */
    virtual bool is_dirty(Attribute attribute) const noexcept
    {
        return dirty_[int(attribute)];
    }

    /**
     * \return First representation ID that is available
     *
     * \param representations   Representation list
     *
     * Example:
     * If the last invoked functions were
     *
     * \code
     * diagonal_covariance(my_diagonal);
     * my_covariance = covariance();
     * \endcode
     *
     * Now, the representation is set to \c DiagonalCovarianceMatrix and
     * \c CovarianceMatrix since \c diagonal_covariance() was used to set the
     * covariance matrix followed by requesting \c covariance().
     * The following subsequent call
     *
     * \code
     * Attribute att = SelectRepresentation({SquareRoot,
     *                                       DiagonalCovarianceMatrix,
     *                                       CovarianceMatrix});
     * \endcode
     *
     * will assign att to DiagonalCovarianceMatrix since that is the first
     * available representation within the initializer-list
     * <tt>{#SquareRoot, #DiagonalCovarianceMatrix, #CovarianceMatrix}</tt>.
     *
     * This method is used to determine the best suitable representation
     * for conversion. It is recommanded to put the diagonal forms at the
     * beginning of the initialization-list. Diagonal forms can be converted
     * most efficiently other  representations.
     */
    virtual Attribute select_first_representation(
            const std::vector<Attribute>& representations) const noexcept
    {
        for (auto& rep: representations)  if (!is_dirty(rep)) return rep;
        return Attributes;
    }

    /**
     * \brief has_full_rank check implementation
     */
    virtual bool has_full_rank(const DiagonalSecondMoment& mat) const
    {
        bool full_rank = true;

        const auto& diag = mat.diagonal();

        for (int i = 0; i < diag.size(); ++i)
        {
            if (std::fabs(diag(i)) < 1e-24)
            {
                full_rank = false;
                break;
            }
        }

        return full_rank;
    }
    /** \endcond */

protected:
    /** \cond internal */
    Variate mean_;                            /**< \brief first moment vector */
    mutable DiagonalSecondMoment covariance_; /**< \brief cov. form */
    mutable DiagonalSecondMoment precision_;  /**< \brief cov. inverse form */
    mutable DiagonalSecondMoment square_root_;/**< \brief cov. square root  */
    mutable bool full_rank_;                  /**< \brief full rank flag */
    mutable Real log_norm_;                   /**< \brief log normalizing const */
    mutable Real determinant_;         /**< \brief determinant of covariance */
    mutable std::vector<bool> dirty_;         /**< \brief data validity flags */
    /** \endcond */
};

/** \} */

}
