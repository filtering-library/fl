/*
 * This is part of the fl library, a C++ Bayesian filtering library
 * (https://github.com/filtering-library)
 *
 * Copyright (c) 2015 Max Planck Society,
 * 				 Autonomous Motion Department,
 * 			     Institute for Intelligent Systems
 *
 * This Source Code Form is subject to the terms of the MIT License (MIT).
 * A copy of the license can be found in the LICENSE file distributed with this
 * source code.
 */

/**
 * @date 2015
 * @author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#pragma once


#include <fl/util/traits.hpp>
#include <fl/filter/filter_interface.hpp>
#include <fl/distribution/discrete_distribution.hpp>
#include <fl/distribution/standard_gaussian.hpp>

namespace fl
{

/**
 * \defgroup particle_filter Particle Filter
 * \ingroup filters
 */

// Particle filter forward declaration
template <typename...> class ParticleFilter;

/**
 * \internal
 * \ingroup particle_filter
 *
 * ParticleFilter Traits
 */
template <
    typename TransitionFunction,
    typename SensorDensity
>
struct Traits<ParticleFilter<TransitionFunction, SensorDensity>>
{
    typedef typename TransitionFunction::State State;
    typedef typename TransitionFunction::Input Input;
    typedef typename SensorDensity::Obsrv      Obsrv;
    typedef DiscreteDistribution<State>             Belief;
};

/**
 * \ingroup particle_filter
 *
 * \brief Represents the general particle filter
 */
template<
    typename TransitionFunction,
    typename SensorDensity
>
class ParticleFilter<TransitionFunction, SensorDensity>
    : public FilterInterface<
                 ParticleFilter<TransitionFunction, SensorDensity>>
{
private:
    /** \cond internal */
    typedef typename TransitionFunction::Noise StateNoise;
    typedef typename SensorDensity::Noise      ObsrvNoise;
    /** \endcond */

public:
    typedef typename TransitionFunction::State     State;
    typedef typename TransitionFunction::Input     Input;
    typedef typename SensorDensity::Obsrv          Obsrv;
    typedef DiscreteDistribution<State>                 Belief;

public:
    ParticleFilter(const TransitionFunction& process_model,
                   const SensorDensity& obsrv_model,
                   const Real& max_kl_divergence = 1.0)
        : process_model_(process_model),
          obsrv_model_(obsrv_model),
          process_noise_(process_model.noise_dimension()),
          obsrv_noise_(obsrv_model.noise_dimension()),
          max_kl_divergence_(max_kl_divergence)
    { }

    /**
     * \brief Overridable default destructor
     */
    virtual ~ParticleFilter() noexcept { }

    /**
     * \copydoc FilterInterface::predict
     */
    virtual void predict(const Belief& prior_belief,
                         const Input& input,
                         Belief& predicted_belief)
    {
        predicted_belief = prior_belief;
        for(int i = 0; i < predicted_belief.size(); i++)
        {
            predicted_belief.location(i) =
                    process_model_.state(prior_belief.location(i),
                                         process_noise_.sample(),
                                         input);
        }
    }

    /**
     * \copydoc FilterInterface::update
     */
    virtual void update(const Belief& predicted_belief,
                        const Obsrv& obsrv,
                        Belief& posterior_belief)
    {
        // if the samples are too concentrated then resample
        if(predicted_belief.kl_given_uniform() > max_kl_divergence_)
        {
            posterior_belief.from_distribution(predicted_belief,
                                               predicted_belief.size());
        }
        else
        {
            posterior_belief = predicted_belief;
        }

        // update the weights of the particles with the likelihoods
        posterior_belief.delta_log_prob_mass(
             obsrv_model_.log_probabilities(obsrv, predicted_belief.locations()));
    }

    /**
     * \copydoc FilterInterface::predict_and_update
     */
    virtual void predict_and_update(const Belief& prior_belief,
                                    const Input& input,
                                    const Obsrv& observation,
                                    Belief& posterior_belief)
    {
        predict(prior_belief, input, posterior_belief);
        update(posterior_belief, observation, posterior_belief);
    }

public: /* factory functions */
    virtual Belief create_belief() const
    {
        auto belief = Belief(process_model().state_dimension());
        return belief;
    }

public: /* accessors */
    TransitionFunction& process_model()
    {
        return process_model_;
    }
    SensorDensity& obsrv_model()
    {
        return obsrv_model_;
    }

    const TransitionFunction& process_model() const
    {
        return process_model_;
    }

    const SensorDensity& obsrv_model() const
    {
        return obsrv_model_;
    }

    virtual std::string name() const
    {
        return "ParticleFilter<"
                + this->list_arguments(
                            process_model().name(),
                            obsrv_model().name())
                + ">";
    }

    virtual std::string description() const
    {
        return "Non-parametric ParticleFilter with"
                + this->list_descriptions(
                            process_model().description(),
                            obsrv_model().description());
    }
protected:
    TransitionFunction process_model_;
    SensorDensity obsrv_model_;

    StandardGaussian<StateNoise> process_noise_;
    StandardGaussian<ObsrvNoise> obsrv_noise_;

    /**
     * when the KL divergence KL(p||u), where p is the particle distribution
     * and u is the uniform distribution, exceeds max_kl_divergence_, then there
     * is a resampling step. can be understood as -log(f) where f is the
     * fraction of nonzero particles.
     */
    fl::Real max_kl_divergence_;
};

}
