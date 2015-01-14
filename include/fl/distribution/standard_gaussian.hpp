/**
 * \file standard_gaussian.hpp
 * \date May 2014
 * @author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * @author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__DISTRIBUTION__STANDARD_GAUSSIAN_HPP
#define FL__DISTRIBUTION__STANDARD_GAUSSIAN_HPP

#include <Eigen/Dense>

#include <random>

#include <fl/util/random_seed.hpp>
#include <fl/util/traits.hpp>
#include <fl/distribution/interface/sampling.hpp>
#include <fl/exception/exception.hpp>

namespace fl
{

/**
 * \ingroup distributions
 */
template <typename Vector>
class StandardGaussian:
        public Sampling<Vector>
{
public:
    explicit StandardGaussian(const int dimension = DimensionOf<Vector>())
        : dimension_ (dimension),
          generator_(RANDOM_SEED),
          gaussian_distribution_(0.0, 1.0),
          gaussian_generator_(std::bind(gaussian_distribution_, generator_))
    {
    }

    virtual ~StandardGaussian() { }

    virtual Vector Sample()
    {
        Vector gaussian_sample(Dimension());
        for (int i = 0; i < Dimension(); i++)
        {
            gaussian_sample(i) = gaussian_generator_();
        }

        return gaussian_sample;
    }

    virtual int Dimension() const
    {
        return dimension_;
    }

    virtual void Dimension(size_t new_dimension)
    {
        if (dimension_ == new_dimension) return;

        if (fl::IsFixed<Vector::SizeAtCompileTime>())
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
    std::mt19937 generator_;
    std::normal_distribution<> gaussian_distribution_;
    std::function<double()> gaussian_generator_;
};

// specialization for scalar
template<>
class StandardGaussian<double>: public Sampling<double>
{
public:
    StandardGaussian()
        : generator_(RANDOM_SEED),
          gaussian_distribution_(0.0, 1.0),
          gaussian_generator_(std::bind(gaussian_distribution_, generator_))
    { }

    virtual ~StandardGaussian() { }

    virtual double Sample()
    {
        return gaussian_generator_();
    }

    virtual int Dimension() const
    {
        return 1;
    }

private:
    std::mt19937 generator_;
    std::normal_distribution<> gaussian_distribution_;
    std::function<double()> gaussian_generator_;
};

}

#endif
