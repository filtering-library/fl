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
 * \file damped_wiener_process_model.hpp
 * \date 05/25/2014
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__MODEL__PROCESS__DAMPED_WIENER_PROCESS_MODEL_HPP
#define FL__MODEL__PROCESS__DAMPED_WIENER_PROCESS_MODEL_HPP

#include <Eigen/Dense>

#include <fl/util/assertions.hpp>
#include <fl/distribution/gaussian.hpp>
#include <fl/model/process/process_model_interface.hpp>

namespace fl
{

// Forward declarations
template <typename State> class DampedWienerProcessModel;

/**
 * DampedWienerProcess process model traits. This trait definition contains
 * all types used internally within the linear model. Additionally, it provides
 * the types needed externally to use the model.
 */
template <typename State_>
struct Traits<DampedWienerProcessModel<State_>>
{
    /*
     * Required types
     *
     * - Scalar
     * - State
     * - Input
     * - Noise
     */
    typedef State_ State;
    typedef typename State::Scalar Scalar;
    typedef Eigen::Matrix<Scalar, State::SizeAtCompileTime, 1>  Input;
    typedef Eigen::Matrix<Scalar, State::SizeAtCompileTime, 1>  Noise;

    /* Model interfaces*/
    /**
     * \brief Basic process model interface type defined using the State, Noise
     *        and Input types of this process
     */
    typedef ProcessModelInterface<State, Noise, Input> ProcessInterfaceBase;

    /**
     * \brief Interface definition used to map standard normal noise variates
     *        onto a state variate
     */
    typedef StandardGaussianMapping<State, Noise> GaussianMappingBase;

    /** \cond INTERNAL */
    typedef Gaussian<Noise> NoiseGaussian;
    typedef typename NoiseGaussian::SecondMoment SecondMoment;
    /** \endcond */
};

/**
 * \ingroup process_models
 * \sa Traits<DampedWienerProcessModel<State>>
 * \todo What does IntegratedDampedWienerProcessModel represent?
 * \todo missing unit tests
 *
 * \brief Represents ...
 *
 * \details
 */
template <typename State>
class DampedWienerProcessModel
        : public Traits<DampedWienerProcessModel<State>>::GaussianMappingBase,
          public Traits<DampedWienerProcessModel<State>>::ProcessInterfaceBase
{
private:
    /** Typdef of \c This for #from_traits(TypeName) helper */
    typedef DampedWienerProcessModel<State> This;

public:
    typedef from_traits(Scalar);
    typedef from_traits(SecondMoment);
    typedef from_traits(Input);
    typedef from_traits(Noise);
    typedef from_traits(NoiseGaussian);

public:

    /**
     * Constructor of a fixed-size and dynamic-size DampedWiener
     * process model.
     *
     * \param dim   State dimension which must be specified for dynamic-size
     *              states. If the state has a compile-time fix size, the \c dim
     *              argument is deduced automatically
     */
    explicit
    DampedWienerProcessModel(int dim = DimensionOf<State>())
        : Traits<This>::GaussianMappingBase(dim),
          gaussian_(dim)
    {
        static_assert(IsDynamic<State::SizeAtCompileTime>() ||
                      DimensionOf<State>() > 0,
                      "Static state dimension must be greater than zero.");
    }

    /**
     * \brief  Overridable default destructor
     */
    virtual ~DampedWienerProcessModel() { }

    /**
     * Mapps the specified noise sample onto the a state variate.
     *
     * \param sample    Noise variate
     *
     * \return A state variate given the noise sample
     */
    virtual State map_standard_normal(const Noise& sample) const
    {
        return gaussian_.map_standard_normal(sample);
    }

    /**
     * \copydoc ProcessModelInterface::condition
     */
    virtual void condition(const Scalar&  delta_time,
                           const State&  state,
                           const Input&   input)
    {
        gaussian_.mean(mean(delta_time, state, input));
        gaussian_.diagonal_covariance(covariance(delta_time));
    }


    /* interface API */
    /**
     * \copydoc ProcessModelInterface::condition
     */
    virtual State predict_state(double delta_time,
                                const State& state,
                                const Noise& noise,
                                const Input& input)
    {
        condition(delta_time, state, input);

        return map_standard_normal(noise);
    }

    virtual void parameters(const Scalar& damping,
                            const SecondMoment& noise_covariance)
    {
        damping_ = damping;
        noise_covariance_ = noise_covariance;
    }

    /**
     * This is the same as the state dimension since the state represents the
     * process distribution variate.
     *
     * \return dimension of the state or the process variate
     */
    virtual int dimension() const
    {
        return state_dimension();
    }

    /**
     * \copydoc ProcessModelInterface::state_dimension
     */
    virtual int state_dimension() const
    {
        return this->standard_variate_dimension();
    }

    /**
     * \copydoc ProcessModelInterface::noise_dimension
     */
    virtual int noise_dimension() const
    {
        return this->standard_variate_dimension();
    }

    /**
     * \copydoc ProcessModelInterface::input_dimension
     */
    virtual int input_dimension() const
    {
        return this->standard_variate_dimension();
    }

protected:
    /**
     * \param delta_time    \f$\Delta t\f$
     * \param state         \f$x\f$
     * \param input         \f$u\f$
     *
     * \return
     * \f[
     * \mu = e^{-\Delta t d} x + \frac{1-e^{-\Delta t d}}{d}u
     * \f]
     */
    State mean(const Scalar& delta_time,
               const State& state,
               const Input& input)
    {
        // for readability ... the compiler optimizes this out
        const double dt = delta_time;
        const Scalar d = damping_;
        const double exp_ddt = std::exp(-d * dt);

        if(d == 0) return state + delta_time * input;

        State state_expectation = (1.0 - exp_ddt) / d * input + exp_ddt * state;

        /*
         * if the damping_ is too small, the result might be nan, we thus return
         * the limit for damping_ -> 0
         */
        if(!std::isfinite(state_expectation.norm()))
        {
            state_expectation = state + d * input;
        }

        return state_expectation;
    }

    /**
     * \param delta_time    \f$\Delta t\f$
     *
     * \return
     * \f[
     * \Sigma = f \Sigma_{v}
     * \f]
     *
     * with
     *
     * noise covariance matrix \f$\Sigma_{v}\f$ and the factor
     * \f[
     * f = \frac{1 - e^{-2 \Delta t d}}{2 d}
     * \f]
     * while
     * \f$d\f$ being the damping constant.
     */
    SecondMoment covariance(const Scalar& delta_time)
    {
        // for readability ... the compiler optimizes this out
        const double dt = delta_time;
        const Scalar d = damping_;

        if(d == 0) return dt * noise_covariance_;

        Scalar factor = (1.0 - std::exp(-2.0 * d * dt)) / (2.0 * d);

        if(!std::isfinite(factor)) factor = dt;

        return factor * noise_covariance_;
    }

private:
    // conditional
    NoiseGaussian gaussian_;

    // parameters
    Scalar damping_;
    SecondMoment noise_covariance_;
};

}

#endif
