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
 * \file gaussian.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#ifndef FL__DISTRIBUTION__GAUSSIAN_HPP
#define FL__DISTRIBUTION__GAUSSIAN_HPP

#include <Eigen/Dense>

#include <vector>
#include <string>
#include <type_traits>

#include <fl/util/traits.hpp>
#include <fl/distribution/interface/moments.hpp>
#include <fl/distribution/interface/evaluation.hpp>
#include <fl/distribution/interface/gaussian_map.hpp>
#include <fl/exception/exception.hpp>

namespace fl
{

// Forward declarations
template <typename Vector> class Gaussian;

/**
 * Gaussian distribution traits. This trait definition contains all types used
 * internally within the distribution. Additionally, it provides the types
 * needed externally to use the Gaussian.
 */
template <typename Vector_>
struct Traits<Gaussian<Vector_>>
{    
    enum
    {
        /**
         * \brief Gaussian dimension
         *
         * For fixed-size Point type and hence a fixed-size Gaussian, the
         * \c Dimension value is greater zero. Dynamic-size Gaussians have the
         * dymension Eigen::Dynamic.
         */
        Dimension = Vector_::SizeAtCompileTime
    };

    /**
     * \brief Gaussian variable type
     */
    typedef Vector_ Vector;

    /**
     * \brief Internal scalar type (e.g. double, float, std::complex)
     */
    typedef typename Vector::Scalar Scalar;

    /**
     * \brief Random variable type. The Noise type is used in mapping of noise
     * samples into the current Gaussian space.
     */
    typedef Eigen::Matrix<Scalar, Dimension, 1> Noise;

    /**
     * \brief Second moment type
     */
    typedef Eigen::Matrix<Scalar, Dimension, Dimension> Operator;

    /**
     * \brief Moments interface of a Gaussian
     */
    typedef Moments<Vector, Operator> MomentsBase;

    /**
     * \brief Evalluation interface of a Gaussian
     */
    typedef Evaluation<Vector, Scalar> EvaluationBase;

    /**
     * \brief GaussianMap interface of a Gaussian
     */
    typedef GaussianMap<Vector, Noise> GaussianMapBase;
};

/**
 * \ingroup exceptions
 *
 * Exception used in case of accessing dynamic-size distribution attributes
 * without initializing the Gaussian using a dimension greater 0.
 */
class GaussianUninitializedException:
    public fl::Exception
{
public:
    /**
     * Creates a GaussianUninitializedException
     */
    GaussianUninitializedException():
        fl::Exception("Accessing uninitialized dynamic-size distribution. "
                      "Gaussian dimension is 0. "
                      "Use ::Dimension(dimension) to initialize the "
                      "distribution!") { }
};

/**
 * \ingroup exceptions
 *
 * Exception representing a unsupported representation ID
 */
class InvalidGaussianRepresentationException:
    public fl::Exception
{
public:
    /**
     * Creates an InvalidGaussianRepresentationException
     */
    InvalidGaussianRepresentationException():
        fl::Exception("Invalid Gaussian covariance representation") { }
};

/**
 * \class Gaussian
 *
 * \brief General Gaussian Distribution
 * \ingroup distributions
 * @{
 *
 * The Gaussian is a general purpose distribution. It can be used in various
 * ways while maintaining efficienty at the same time. This is due to it's
 * multi-representation structure. The distribution can be represented either by
 *
 *  - the covariance matrix,
 *  - the precision matrix,
 *  - the covariance square root matrix (Cholesky decomposition or LDLT),
 *  - or the diagonal form of the previous three options.
 *
 * A change in one representation results in change of all other
 * representations.
 *
 * Two key features of the distribution are its aibility to evaluation the
 * probability of a given sample and to map a noise sample into the distribution
 * sample space.
 *
 * \cond INTERNAL
 * The Gaussian internal structure uses lazy assignments or write on read
 * technique. Due to the multi-representation of the Gaussian, modifying one
 * representation affects all remaining ones. If one of the representation is
 * modified, the other representations are only then updated when needed. This
 * minimizes redundant computation and increases efficienty.
 * \endcond
 */
template <typename Vector_>
class Gaussian:
        public Traits<Gaussian<Vector_>>::MomentsBase,
        public Traits<Gaussian<Vector_>>::EvaluationBase,
        public Traits<Gaussian<Vector_>>::GaussianMapBase
{
public:
    typedef Gaussian<Vector_> This;

    typedef typename Traits<This>::Vector     Vector;
    typedef typename Traits<This>::Scalar     Scalar;
    typedef typename Traits<This>::Operator   Operator;
    typedef typename Traits<This>::Noise      Noise;

    using Traits<This>::GaussianMapBase::NoiseDimension;

protected:
    /** \cond INTERNAL */
    /**
     * \enum Attribute
     * Implementation attributes. The enumeration lists the different
     * representations along with other properties such as the rank of the
     * second moment and the log normalizer.
     */
    enum Attribute
    {
        CovarianceMatrix = 0,     /**< Covariance mat. */
        PrecisionMatrix,          /**< Inverse of the cov. mat. */
        SquareRootMatrix,         /**< Cholesky decomp. of the cov. mat. */
        DiagonalCovarianceMatrix, /**< Diagonal form of the of cov. mat. */
        DiagonalPrecisionMatrix,  /**< Diagonal form of the inv cov. mat. */
        DiagonalSquareRootMatrix, /**< Diagonal form of the Cholesky decomp. */
        Rank,                     /**< Covariance Rank */
        Normalizer,               /**< Log probability normalizer */

        Attributes                /**< Total number of attribute */
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
    explicit Gaussian(const unsigned& dimension = DimensionOf<Vector>()):
        Traits<This>::GaussianMapBase(dimension),
        dirty_(Attributes, true)
    {
        static_assert(Vector::SizeAtCompileTime != 0,
                      "Illegal static dimension");
        SetStandard();
    }

    /**
     * \brief Overridable default constructor
     */
    virtual ~Gaussian() { }

    /**
     * \return Gaussian dimension
     */
    virtual int Dimension() const
    {
        return NoiseDimension();
    }

    /**
     * \return Gaussian first moment
     */
    virtual Vector Mean() const
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
     * \cond INTERNAL
     * \pre |{Valid Representations}| > 0
     * \post {Valid Representations}
     *       = {Valid Representations} \f$ \cup \f$ {#CovarianceMatrix}
     * \endcond
     */
    virtual Operator Covariance() const
    {
        if (Dimension() == 0)
        {
            BOOST_THROW_EXCEPTION(GaussianUninitializedException());
        }

        if (is_dirty(CovarianceMatrix) && is_dirty(DiagonalCovarianceMatrix))
        {
            switch (select_first_representation({DiagonalSquareRootMatrix,
                                                 DiagonalPrecisionMatrix,
                                                 SquareRootMatrix,
                                                 PrecisionMatrix}))
            {
            case SquareRootMatrix:
                covariance_ = square_root_ * square_root_.transpose();
                break;

            case PrecisionMatrix:
                covariance_ = precision_.inverse();
                break;

            case DiagonalSquareRootMatrix:
                covariance_.setZero(Dimension(), Dimension());
                for (size_t i = 0; i < square_root_.diagonalSize(); ++i)
                {
                    covariance_(i, i) = square_root_(i, i) * square_root_(i, i);
                }
                break;

            case DiagonalPrecisionMatrix:
                covariance_.setZero(Dimension(), Dimension());
                for (size_t i = 0; i < precision_.diagonalSize(); ++i)
                {
                    covariance_(i, i) = 1./precision_(i, i);
                }
                break;

            default:
                BOOST_THROW_EXCEPTION(InvalidGaussianRepresentationException());
                break;
            }

            updated_internally(CovarianceMatrix);
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
     * \cond INTERNAL
     * \pre |{Valid Representations}| > 0
     * \post {Valid Representations}
     *       = {Valid Representations} \f$ \cup \f$ {#PrecisionMatrix}
     * \endcond
     */
    virtual const Operator& Precision() const
    {
        if (Dimension() == 0)
        {
            BOOST_THROW_EXCEPTION(GaussianUninitializedException());
        }

        if (is_dirty(PrecisionMatrix) && is_dirty(DiagonalPrecisionMatrix))
        {
            const Operator& cov = Covariance();

            switch (select_first_representation({DiagonalCovarianceMatrix,
                                                 DiagonalSquareRootMatrix,
                                                 CovarianceMatrix,
                                                 SquareRootMatrix}))
            {
            case CovarianceMatrix:
            case SquareRootMatrix:
                precision_ = Covariance().inverse();
                break;

            case DiagonalCovarianceMatrix:
            case DiagonalSquareRootMatrix:
                precision_.setZero(Dimension(), Dimension());
                for (size_t i = 0; i < cov.rows(); ++i)
                {
                    precision_(i, i) = 1./cov(i, i);
                }
                break;

            default:
                BOOST_THROW_EXCEPTION(InvalidGaussianRepresentationException());
                break;
            }

            updated_internally(PrecisionMatrix);
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
     * \cond INTERNAL
     * \pre |{Valid Representations}| > 0
     * \post {Valid Representations}
     *       = {Valid Representations} \f$ \cup \f$ {#SquareRootMatrix}
     * \endcond
     */
    virtual const Operator& SquareRoot() const
    {
        if (Dimension() == 0)
        {
            BOOST_THROW_EXCEPTION(GaussianUninitializedException());
        }

        if (is_dirty(SquareRootMatrix) && is_dirty(DiagonalSquareRootMatrix))
        {
            const Operator& cov = Covariance();

            switch (select_first_representation({DiagonalCovarianceMatrix,
                                                 DiagonalPrecisionMatrix,
                                                 CovarianceMatrix,
                                                 PrecisionMatrix}))
            {
            case CovarianceMatrix:
            case PrecisionMatrix:
            {
                Eigen::LDLT<Operator> ldlt;
                ldlt.compute(Covariance());
                Vector D_sqrt = ldlt.vectorD();
                for(size_t i = 0; i < D_sqrt.rows(); ++i)
                {
                    D_sqrt(i) = std::sqrt(std::fabs(D_sqrt(i)));
                }
                square_root_ = ldlt.transpositionsP().transpose()
                                * (Operator)ldlt.matrixL()
                                * D_sqrt.asDiagonal();
            } break;

            case DiagonalCovarianceMatrix:
            case DiagonalPrecisionMatrix:
            {
                square_root_.setZero(Dimension(), Dimension());
                for (size_t i = 0; i < square_root_.rows(); ++i)
                {
                    square_root_(i, i) = std::sqrt(cov(i, i));
                }
            } break;

            default:
                BOOST_THROW_EXCEPTION(InvalidGaussianRepresentationException());
                break;
            }

            updated_internally(SquareRootMatrix);
        }

        return square_root_;
    }

    /**
     * \return True if the covariance matrix has a full rank
     *
     * \throws see Covariance()
     *
     * \cond INTERNAL
     * \pre |{Valid Representations}| > 0
     * \post {Valid Representations}
     *       = {Valid Representations} \f$ \cup \f$ {#CovarianceMatrix}
     * \endcond
     */
    virtual bool HasFullRank() const
    {
        if (is_dirty(Rank))
        {
            full_rank_ =
               Covariance().colPivHouseholderQr().rank() == Covariance().rows();

            updated_internally(Rank);
        }

        return full_rank_;
    }

    /**
     * @return Log normalizing constant
     *
     * \throws see HasFullRank()
     *
     * \cond INTERNAL
     * \pre |{Valid Representations}| > 0
     * \post {Valid Representations}
     *       = {Valid Representations} \f$ \cup \f$ {#CovarianceMatrix}
     * \endcond
     */
    virtual Scalar LogNormalizer() const
    {
        if (is_dirty(Normalizer))
        {
            if (HasFullRank())
            {
                log_normalizer_ = -0.5
                        * (log(Covariance().determinant())
                           + double(Covariance().rows()) * log(2.0 * M_PI));
            }
            else
            {
                log_normalizer_ = 0.0; // FIXME
            }

            updated_internally(Normalizer);
        }

        return log_normalizer_;
    }

    /**
     * @return Log of the probability of the given sample \c vector
     *
     * @param vector sample which should be evaluated
     *
     * \throws see HasFullRank()
     *
     * \cond INTERNAL
     * \pre |{Valid Representations}| > 0
     * \post {Valid Representations}
     *       = {Valid Representations} \f$ \cup \f$ {#PrecisionMatrix}
     * \endcond
     */
    virtual Scalar LogProbability(const Vector& vector) const
    {
        if(HasFullRank())
        {
            return LogNormalizer() - 0.5
                    * (vector - Mean()).transpose()
                    * Precision()
                    * (vector - Mean());
        }

        return -std::numeric_limits<Scalar>::infinity();
    }

    /**
     * @return a Gaussian sample of the type \c Vector determined by mapping a
     * noise sample into the Gaussian sample space
     *
     * @param sample    Noise Sample
     *
     * \throws see SquareRoot()
     *
     * \cond INTERNAL
     * \pre |{Valid Representations}| > 0
     * \post {Valid Representations}
     *       = {Valid Representations} \f$ \cup \f$ {#SquareRootMatrix}
     * \endcond
     */
    virtual Vector MapStandardGaussian(const Noise& sample) const
    {
        return Mean() + SquareRoot() * sample;
    }

    /**
     * Sets the Gaussian to a standard distribution with zero mean and identity
     * covariance.
     *
     * \cond INTERNAL
     * \pre {}
     * \post
     *  - Fully ranked covariance
     *  - {Valid representations} = {#CovarianceMatrix}
     * \endcond
     */
    virtual void SetStandard()
    {
        mean_.resize(Dimension());
        covariance_.resize(Dimension(), Dimension());

        Mean(Vector::Zero(Dimension()));
        Covariance(Operator::Identity(Dimension(), Dimension()));

        full_rank_ = true;
        updated_internally(Rank);
    }

    /**
     * Changes the dimension of the dynamic-size Gaussian and sets it to a
     * standard distribution with zero mean and identity covariance.
     *
     * @param new_dimension New dimension of the Gaussian
     *
     * \cond INTERNAL
     * \pre {}
     * \post
     *  - Fully ranked covariance
     *  - {Valid representations} = {#CovarianceMatrix}
     * \endcond
     *
     * \throws ResizingFixedSizeEntityException
     *         see GaussianMap::NoiseDimension(size_t)
     */
    virtual void Dimension(size_t new_dimension)
    {
        NoiseDimension(new_dimension);
        SetStandard();
    }

    /**
     * Sets the mean
     *
     * @param mean New Gaussian mean
     *
     * \throws WrongSizeException
     */
    virtual void Mean(const Vector& mean) noexcept
    {
        if (mean_.size() != mean.size())
        {
            BOOST_THROW_EXCEPTION(fl::WrongSizeException(mean.size(),
                                                         mean_.size()));
        }

        mean_ = mean;
    }

    /**
     * Sets the covariance matrix
     * @param covariance New covariance matrix
     *
     * \cond INTERNAL
     * \pre |{Valid Representations}| > 0
     * \post {Valid Representations}
     *       = {Valid Representations} \f$ \cup \f$ {#CovarianceMatrix}
     * \endcond
     *
     * \throws WrongSizeException
     */
    virtual void Covariance(const Operator& covariance) noexcept
    {
        if (covariance_.size() != covariance.size())
        {
            BOOST_THROW_EXCEPTION(fl::WrongSizeException(covariance.size(),
                                                         covariance_.size()));
        }

        covariance_ = covariance;
        updated_externally(CovarianceMatrix);
    }

    /**
     * Sets the covariance matrix in the form of its square root (Cholesky f
     * actor)
     *
     * @param square_root New covariance square root
     *
     * \cond INTERNAL
     * \pre |{Valid Representations}| > 0
     * \post {Valid Representations}
     *       = {Valid Representations} \f$ \cup \f$ {#SquareRootMatrix}
     * \endcond
     *
     * \throws WrongSizeException
     */
    virtual void SquareRoot(const Operator& square_root) noexcept
    {
        if (square_root_.size() != square_root.size())
        {
            BOOST_THROW_EXCEPTION(fl::WrongSizeException(square_root.size(),
                                                         square_root_.size()));
        }

        square_root_ = square_root;
        updated_externally(SquareRootMatrix);
    }

    /**
     * Sets the covariance matrix in the precision form (inverse of covariance)
     *
     * @param precision New precision matrix
     *
     * \cond INTERNAL
     * \pre |{Valid Representations}| > 0
     * \post {Valid Representations}
     *       = {Valid Representations} \f$ \cup \f$ {#PrecisionMatrix}
     * \endcond
     *
     * \throws WrongSizeException
     */
    virtual void Precision(const Operator& precision) noexcept
    {
        if (precision_.size() != precision.size())
        {
            BOOST_THROW_EXCEPTION(fl::WrongSizeException(precision.size(),
                                                         precision_.size()));
        }

        precision_ = precision;
        updated_externally(PrecisionMatrix);
    }    

    /**
     * Sets the covariance matrix as a diagonal matrix
     *
     * @param diag_covariance New diagonal covariance matrix
     *
     * \cond INTERNAL
     * \pre |{Valid Representations}| > 0
     * \post {Valid Representations}
     *       = {Valid Representations} \f$ \cup \f$ {#DiagonalCovarianceMatrix}
     * \endcond
     *
     * \throws WrongSizeException
     */
    virtual void DiagonalCovariance(const Operator& diag_covariance) noexcept
    {
        if (diag_covariance.size() != covariance_.size())
        {
            BOOST_THROW_EXCEPTION(
                fl::WrongSizeException(
                    diag_covariance.size(), covariance_.size()));
        }

        covariance_ = diag_covariance.diagonal().asDiagonal();
        updated_externally(DiagonalCovarianceMatrix);
    }

    /**
     * Sets the covariance matrix in its diagonal square root form
     *
     * @param diag_square_root New diagonal square root of the covariance
     *
     * \cond INTERNAL
     * \pre |{Valid Representations}| > 0
     * \post {Valid Representations}
     *       = {Valid Representations} \f$ \cup \f$ {#DiagonalSquareRootMatrix}
     * \endcond
     *
     * \throws WrongSizeException
     */
    virtual void DiagonalSquareRoot(const Operator& diag_square_root) noexcept
    {
        if (diag_square_root.size() != square_root_.size())
        {
            BOOST_THROW_EXCEPTION(
                fl::WrongSizeException(
                    diag_square_root.size(), square_root_.size()));
        }

        square_root_ = diag_square_root.diagonal().asDiagonal();
        updated_externally(DiagonalSquareRootMatrix);
    }

    /**
     * Sets the covariance matrix in its diagonal precision form
     *
     * @param diag_precision New diagonal precision matrix
     *
     * \cond INTERNAL
     * \pre |{Valid Representations}| > 0
     * \post {Valid Representations}
     *       = {Valid Representations} \f$ \cup \f$ {#DiagonalPrecisionMatrix}
     * \endcond
     *
     * \throws WrongSizeException
     */
    virtual void DiagonalPrecision(const Operator& diag_precision) noexcept
    {
        if (diag_precision.size() != precision_.size())
        {
            BOOST_THROW_EXCEPTION(
                fl::WrongSizeException(
                    diag_precision.size(), precision_.size()));
        }

        precision_ = diag_precision.diagonal().asDiagonal();
        updated_externally(DiagonalPrecisionMatrix);
    }

protected:
    /** \cond INTERNAL */
    /**
     * Flags the specified attribute as valid and the rest of attributes as
     * dirty.
     *
     * @param attribute Modified attribute
     */
    virtual void updated_externally(Attribute attribute) const noexcept
    {
        std::fill(dirty_.begin(), dirty_.end(), true);
        updated_internally(attribute);
    }

    /**
     * Flags the specified attribute as valid.
     *
     * @param attribute Modified attribute
     */
    virtual void updated_internally(Attribute attribute) const noexcept
    {
        dirty_[attribute] = false;
    }

    /**
     * @return True if any of the other representation was modified.
     * @param attribute     Attribute in question
     */
    virtual bool is_dirty(Attribute attribute) const noexcept
    {
        return dirty_[int(attribute)];
    }

    /**
     * @return First representation ID that is available
     *
     * @param representations   Representation list
     *
     * Example:
     * If the last invoked functions were
     *
     * \code
     * DiagonalCovariance(my_diagonal);
     * my_covariance = Covariance();
     * \endcode
     *
     * Now, the representation is set to \c DiagonalCovarianceMatrix and
     * \c CovarianceMatrix since \c DiagonalCovariance() was used to set the
     * covariance matrix followed by requesting \c Covariance().
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
    /** \endcond */

protected:    
    /** \cond INTERNAL */
    Vector mean_;                     /**< \brief first moment vector */
    mutable Operator covariance_;     /**< \brief cov. form */
    mutable Operator precision_;      /**< \brief cov. inverse form */
    mutable Operator square_root_;    /**< \brief cov. square root form */
    mutable bool full_rank_;          /**< \brief full rank flag */
    mutable Scalar log_normalizer_;   /**< \brief log normalizing constant */
    mutable std::vector<bool> dirty_; /**< \brief data validity flags */
    /** \endcond */
};

/** @} */

}

#endif
