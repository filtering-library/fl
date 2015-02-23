/**
 * \file standard_gaussian.hpp
 * \date May 2014
 * \author Jan Issac (jan.issac@gmail.com)
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#ifndef FL__DISTRIBUTION__STANDARD_GAUSSIAN_HPP
#define FL__DISTRIBUTION__STANDARD_GAUSSIAN_HPP

#include <Eigen/Dense>

#include <random>
#include <type_traits>

#include <fl/util/random.hpp>
#include <fl/util/traits.hpp>
#include <fl/util/math.hpp>
#include <fl/distribution/interface/sampling.hpp>
#include <fl/exception/exception.hpp>

namespace fl
{

/**
 * \ingroup distributions
 */
template <typename StandardVariate>
class StandardGaussian:
        public Sampling<StandardVariate>
{
public:
    explicit StandardGaussian(int dim = DimensionOf<StandardVariate>())
        : dimension_ (dim),
          generator_(fl::seed()),
          gaussian_distribution_(0.0, 1.0)
    {
    }

    virtual ~StandardGaussian() { }

    virtual StandardVariate sample() const
    {
        StandardVariate gaussian_sample(dimension());
        for (int i = 0; i < dimension(); i++)
        {
            gaussian_sample(i) = gaussian_distribution_(generator_);
        }

        return gaussian_sample;
    }

    virtual int dimension() const
    {
        return dimension_;
    }

    virtual void dimension(int new_dimension)
    {
        if (dimension_ == new_dimension) return;

        if (fl::IsFixed<StandardVariate::SizeAtCompileTime>())
        {
            fl_throw(
                fl::ResizingFixedSizeEntityException(dimension_,
                                                     new_dimension,
                                                     "Gaussian"));
        }

        dimension_ = new_dimension;
    }

private:
    int dimension_;
    mutable fl::mt11213b generator_;
    mutable std::normal_distribution<> gaussian_distribution_;
};


/** \cond IMPL_DETAILS */
/**
 * Floating point implementation for Scalar types float, double and long double
 */
template <typename Scalar>
class StandardGaussianFloatingPointScalarImpl
{
    static_assert(
        std::is_floating_point<Scalar>::value,
        "Scalar must be a floating point (float, double, long double)");

public:
    StandardGaussianFloatingPointScalarImpl()
        : generator_(fl::seed()),
          gaussian_distribution_(Scalar(0.), Scalar(1.))
    { }

    Scalar sample_impl() const
    {
        return gaussian_distribution_(generator_);
    }

protected:
    mutable fl::mt11213b generator_;
    mutable std::normal_distribution<Scalar> gaussian_distribution_;
};
/** \endcond */


/**
 * Float floating point StandardGaussian specialization
 * \ingroup distributions
 */
template <>
class StandardGaussian<float>
        : public Sampling<float>,
          public StandardGaussianFloatingPointScalarImpl<float>
{
public:
    /**
     * \copydoc Sampling::sample
     */
    virtual float sample() const
    {
        return this->sample_impl();
    }
};

/**
 * Double floating point StandardGaussian specialization
 * \ingroup distributions
 */
template <>
class StandardGaussian<double>
        : public Sampling<double>,
          public StandardGaussianFloatingPointScalarImpl<double>
{
public:
    /**
     * \copydoc Sampling::sample
     */
    virtual double sample() const
    {
        return this->sample_impl();
    }
};

/**
 * Long double floating point StandardGaussian specialization
 * \ingroup distributions
 */
template <>
class StandardGaussian<long double>
        : public Sampling<long double>,
          public StandardGaussianFloatingPointScalarImpl<long double>
{
public:
    /**
     * \copydoc Sampling::sample
     */
    virtual long double sample() const
    {
        return this->sample_impl();
    }
};

}

#endif
