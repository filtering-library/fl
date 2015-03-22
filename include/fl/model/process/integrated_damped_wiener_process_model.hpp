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
 * \file integrated_damped_wiener_process_model.hpp
 * \date 05/25/2014
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__MODEL__PROCESS__INTEGRATED_DAMPED_WIENER_PROCESS_MODEL_HPP
#define FL__MODEL__PROCESS__INTEGRATED_DAMPED_WIENER_PROCESS_MODEL_HPP

#include <fl/util/math.hpp>
#include <fl/util/assertions.hpp>
#include <fl/distribution/gaussian.hpp>
#include <fl/distribution/interface/standard_gaussian_mapping.hpp>
#include <fl/model/process/process_model_interface.hpp>
#include <fl/model/process/damped_wiener_process_model.hpp>

namespace fl
{

// Forward declarations
template <typename State> class IntegratedDampedWienerProcessModel;

/**
 * IntegratedDampedWienerProcess process model traits. This trait definition
 * contains all types used internally within the linear model. Additionally,
 * it provides the types needed externally to use the model.
 */
template <typename State_>
struct Traits<IntegratedDampedWienerProcessModel<State_>>
{
    enum
    {
        /**
         * \brief Degree-of-freedom represents the number of position
         * components, e.g. xyz-coordinates
         */
        DegreeOfFreedom = IsFixed<State_::RowsAtCompileTime>()
                             ? State_::RowsAtCompileTime/2
                             : Eigen::Dynamic
    };

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
    typedef Eigen::Matrix<Scalar, DegreeOfFreedom, 1> Input;
    typedef Eigen::Matrix<Scalar, DegreeOfFreedom, 1> Noise;

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
    typedef Eigen::Matrix<Scalar, DegreeOfFreedom, 1> WienerProcessState;
    typedef DampedWienerProcessModel<WienerProcessState> DampedWienerProcess;

    typedef Gaussian<Noise> NoiseGaussian;
    typedef typename NoiseGaussian::SecondMoment SecondMoment;
    /** \endcond */
};

/**
 * \ingroup process_models
 *
 * \todo What does IntegratedDampedWienerProcessModel represent?
 * \todo missing unit tests
 * \brief Represents ...
 *
 *
 * \details
 */
template <typename State>
class IntegratedDampedWienerProcessModel
        : public Traits<
                    IntegratedDampedWienerProcessModel<State>
                 >::ProcessInterfaceBase,

          public Traits<
                    IntegratedDampedWienerProcessModel<State>
                 >::GaussianMappingBase
{
protected:
    /** \cond INTERNAL */
    typedef IntegratedDampedWienerProcessModel<State> This;
    typedef typename Traits<This>::SecondMoment SecondMoment;
    typedef typename Traits<This>::NoiseGaussian NoiseGaussian;
    typedef typename Traits<This>::DampedWienerProcess DampedWienerProcess;
    /** \endcond */

public:
    /* public conceptual interface types */
    typedef typename Traits<This>::Scalar Scalar;
    typedef typename Traits<This>::Input  Input;
    typedef typename Traits<This>::Noise  Noise;

public:
    /**
     * Constructor of a fixed-size and dynamic-size IntegratedDampedWiener
     * process model.
     *
     * \param dof   Degree-of-freedom which is dim(State)/2 which must be
     *              specified for dynamic-size states. If the state has a
     *              compile-time fix size, the \c dof argument is deduced
     *              automatically
     */
    explicit
    IntegratedDampedWienerProcessModel(int dof = Traits<This>::DegreeOfFreedom)
        : Traits<This>::GaussianMappingBase(dof),
          velocity_distribution_(dof),
          position_distribution_(dof)
    {
        static_assert(IsDynamic<State::SizeAtCompileTime>() ||
                      DimensionOf<State>() > 0,
                      "Static state dimension must be greater than zero.");

        static_assert(DimensionOf<State>() % 2 == 0,
                      "Dimension must be a multitude of 2 or dynamic-size");
    }

    /**
     * \brief  Overridable default destructor
     */
    virtual ~IntegratedDampedWienerProcessModel() { }

    /**
     * Mapps the specified noise sample onto the a state variate. The noise
     * sample is used to determine the new position and the velocity components.
     *
     * \param sample    Noise variate
     *
     * \return A state variate given the noise sample
     */
    virtual State map_standard_normal(const Noise& sample) const
    {
        State state(state_dimension());

        state.topRows(position_distribution_.dimension()) =
            position_distribution_.map_standard_normal(sample);

        state.bottomRows(velocity_distribution_.dimension()) =
            velocity_distribution_.map_standard_normal(sample);

        return state;
    }

    /**
     * \copydoc ProcessModelInterface::condition
     */
    virtual void condition(const Scalar& delta_time,
                           const State& state,
                           const Input& input)
    {
        position_distribution_.mean(
            mean(state.topRows(position_distribution_.dimension()),
                 state.bottomRows(velocity_distribution_.dimension()),
                 input,
                 delta_time));

        position_distribution_.covariance(covariance(delta_time));

        velocity_distribution_.condition(
            delta_time,
            state.bottomRows(velocity_distribution_.dimension()),
            input);
    }

    /**
     * \copydoc ProcessModelInterface::predict_state
     */
    virtual State predict_state(double delta_time,
                                const State& state,
                                const Noise& noise,
                                const Input& input)
    {
        condition(delta_time, state, input);

        return map_standard_normal(noise);
    }

    /**
     * \copydoc ProcessModelInterface::state_dimension
     *
     * The state is composed of a positional and a velocity segments
     * \f$x = [ x_p\ x_v ]^T\f$ with \f$\dim(x_p) = \dim(x_v)\f$. Hence, the
     * state dimension is \f$2 * \dim(x_p)\f$
     */
    virtual int state_dimension() const
    {
        return this->standard_variate_dimension() * 2;
    }

    /**
     * \copydoc ProcessModelInterface::noise_dimension
     *
     * The noise dimension is equal to \f$\dim(x_p)\f$.
     */
    virtual int noise_dimension() const
    {
        return this->standard_variate_dimension();
    }

    /**
     * \copydoc ProcessModelInterface::input_dimension
     *
     * The input dimension is equal to \f$\dim(x_p)\f$.
     */
    virtual int input_dimension() const
    {
        return this->standard_variate_dimension();
    }

    virtual void parameters(
            const double& damping,
            const SecondMoment& acceleration_covariance)
    {
        damping_ = damping;
        acceleration_covariance_ = acceleration_covariance;

        velocity_distribution_.parameters(damping, acceleration_covariance);
    }

protected:
    /**
     * \param position      \f$x\f$
     * \param velocity      \f$\dot{x}\f$
     * \param acceleration  \f$\ddot{x}\f$
     * \param delta_time    \f$\Delta t\f$
     *
     * \return
     * \f[
     * \mu = x
     *       + \frac{1-e^{-\Delta t d}}{d} \dot{x}
     *       + \frac{e^{-\Delta t d}}{d^2} \ddot{x}
     * \f]
     *
     * with damping constant \f$d\f$.
     */
    Input mean(const Input& position,
               const Input& velocity,
               const Input& acceleration,
               const double& delta_time)
    {
        Input mean;

        // for readability ... the compiler optimizes this out
        const double dt = delta_time;
        const Scalar d = damping_;
        const double exp_ddt = std::exp(-d * dt);

        mean = position
                + (exp_ddt + d * dt - 1.0) / std::pow(d, 2) * acceleration
                + (1.0 - exp_ddt) / d  * velocity;

        if(!std::isfinite(mean.norm()))
        {
            mean = position
                    + 0.5 * std::pow(dt, 2) * acceleration
                    + dt * velocity;
        }

        return mean;
    }

    /**
     * \param delta_time    \f$\Delta t\f$
     *
     * \return
     * \f[
     * \Sigma = f \Sigma_{\ddot{x}}
     * \f]
     *
     * with
     *
     * accelaration covariance matrix \f$\Sigma_{\ddot{x}}\f$ and the factor
     * \f[
     * f =  \frac{\gamma
     *            + \int\limits^\infty_{2d\Delta t} \frac{1}{\tau e^{\tau}}d\tau
     *            + \ln(2d\Delta t) + \frac{3}{2}}
     *           {2d} \Delta t^2
     *      + \frac{2-e^{-2d\Delta t}}{4d^2}\Delta t
     *      + \frac{e^{-2d\Delta t}-1}{8d^3}
     * \f]
     * while
     *
     * \f$\gamma\f$ being the Euler-Mascheroni constant, \f$d\f$ the damping
     * constant.
     */
    SecondMoment covariance(const Scalar& delta_time)
    {
        // for readability ... the compiler optimizes this out
        const double dt = delta_time;
        const Scalar d = damping_;

        /*
         * the first argument to the gamma function should be equal to zero,
         * which would not cause the gamma function to diverge as long as the
         * second argument is not zero, which will not be the case.
         */
        Scalar factor =
              (-1.0 + exp(-2.0 * d * dt)) / (8.0 * std::pow(d, 3))
            + ( 2.0 - exp(-2.0 * d * dt)) / (4.0 * std::pow(d, 2)) * dt
            + (-1.5 + fl::GAMMA + fl::igamma(0., 2.0 * d * dt)
            + log(2.0 * d * dt)) / (2.0 * d) * std::pow(dt, 2);


        if(!std::isfinite(factor)) factor = 1.0/3.0 * std::pow(dt, 3);

        return factor * acceleration_covariance_;
    }

private:
    DampedWienerProcess velocity_distribution_;
    NoiseGaussian position_distribution_;

    // model parameters
    Scalar damping_;
    SecondMoment acceleration_covariance_;
};

}

#endif
