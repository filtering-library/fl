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
#include <fl/distribution/interface/moments.hpp>
#include <fl/exception/exception.hpp>

namespace fl
{

/**
 * \ingroup distributions
 */
template <typename StandardVariate>
class StandardGaussian
    : public Sampling<StandardVariate>,
      public Moments<StandardVariate>
{
public:
    typedef StandardVariate Variate;
    typedef typename Moments<StandardVariate>::SecondMoment SecondMoment;

public:
    explicit
    StandardGaussian(int dim = DimensionOf<StandardVariate>())
        : dimension_ (dim),
          generator_(fl::seed()),
          gaussian_distribution_(0.0, 1.0)
    {
    }

    virtual ~StandardGaussian() { }

    virtual StandardVariate sample() const
    {
        StandardVariate gaussian_sample(dimension(), 1);

        for (int i = 0; i < dimension(); i++)
        {
            gaussian_sample(i, 0) = gaussian_distribution_(generator_);
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

    virtual Variate mean() const
    {
        Variate mu;
        mu.setZero(dimension(), 1);
        return mu;
    }

    virtual SecondMoment covariance() const
    {
        SecondMoment cov;
        cov.setIdentity(dimension(), dimension());

        return cov;
    }

private:
    int dimension_;
    mutable fl::mt11213b generator_;
    mutable std::normal_distribution<> gaussian_distribution_;
};

/**
 * Floating point implementation for Scalar types float, double and long double
 */
template <>
class StandardGaussian<Real>
    : public Sampling<Real>,
      public Moments<Real, Real>
{
public:
    StandardGaussian()
        : generator_(fl::seed()),
          gaussian_distribution_(Real(0.), Real(1.))
    { }

    Real sample() const
    {
        return gaussian_distribution_(generator_);
    }

    virtual Real mean() const
    {
        return 0.;
    }

    virtual Real covariance() const
    {
        return 1.;
    }

protected:
    mutable fl::mt11213b generator_;
    mutable std::normal_distribution<Real> gaussian_distribution_;
};

}

#endif
