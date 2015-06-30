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
 * @date 2015
 * @author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * Max-Planck-Institute for Intelligent Systems
 */

#ifndef FL__FILTER__PARTICLE__PARTICLE_FILTER_HPP
#define FL__FILTER__PARTICLE__PARTICLE_FILTER_HPP


#include <fl/distribution/discrete_distribution.hpp>
#include <fl/distribution/standard_gaussian.hpp>


/// TODO: REMOVE UNNECESSARY INCLUDES

#include <map>
#include <tuple>
#include <memory>

#include <fl/util/meta.hpp>
#include <fl/util/traits.hpp>
#include <fl/util/profiling.hpp>

#include <fl/exception/exception.hpp>
#include <fl/filter/filter_interface.hpp>
#include <fl/filter/gaussian/point_set.hpp>
#include <fl/filter/gaussian/feature_policy.hpp>

#include <fl/model/observation/joint_observation_model.hpp>

namespace fl
{

template <typename...> class ParticleFilter;

/**
 * ParticleFilter Traits
 */
template <typename ProcessModel, typename ObservationModel>
struct Traits<ParticleFilter<ProcessModel, ObservationModel>>
{
    typedef typename ProcessModel::State        State;
    typedef typename ProcessModel::Input        Input;
    typedef typename ObservationModel::Obsrv    Obsrv;
    typedef DiscreteDistribution<State>         Belief;
};


template<typename ProcessModel, typename ObservationModel>
class ParticleFilter<ProcessModel, ObservationModel>
    : public FilterInterface<ParticleFilter<ProcessModel, ObservationModel>>
{
private:
    /** \cond INTERNAL */
    typedef typename ProcessModel::Noise      ProcessNoise;
    typedef typename ObservationModel::Noise  ObsrvNoise;
    /** \endcond */

public:
    typedef typename ProcessModel::State        State;
    typedef typename ProcessModel::Input        Input;
    typedef typename ObservationModel::Obsrv    Obsrv;
    typedef DiscreteDistribution<State>         Belief;

public:
    ParticleFilter(const ProcessModel& process_model,
                   const ObservationModel& obsrv_model,
                   const Real& max_kl_divergence = 1.0)
        : process_model_(process_model),
          obsrv_model_(obsrv_model),
          process_noise_(process_model.noise_dimension()),
          obsrv_noise_(obsrv_model.noise_dimension()),
          max_kl_divergence_(max_kl_divergence)
    { }


    /// predict ****************************************************************
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

    /// update *****************************************************************
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

    /// predict and update *****************************************************
    virtual void predict_and_update(const Belief& prior_belief,
                                    const Input& input,
                                    const Obsrv& observation,
                                    Belief& posterior_belief)
    {
        predict(prior_belief, input, posterior_belief);
        update(posterior_belief, observation, posterior_belief);
    }


    /// set and get ************************************************************
    ProcessModel& process_model()
    {
        return process_model_;
    }
    ObservationModel& obsrv_model()
    {
        return obsrv_model_;
    }

    const ProcessModel& process_model() const
    {
        return process_model_;
    }

    const ObservationModel& obsrv_model() const
    {
        return obsrv_model_;
    }

    virtual Belief create_belief() const
    {
        auto belief = Belief(process_model().state_dimension());
        return belief;
    }

protected:
    ProcessModel process_model_;
    ObservationModel obsrv_model_;

    StandardGaussian<ProcessNoise> process_noise_;
    StandardGaussian<ObsrvNoise> obsrv_noise_;

    // when the KL divergence KL(p||u), where p is the particle distribution
    // and u is the uniform distribution, exceeds max_kl_divergence_, then there
    // is a resampling step. can be understood as -log(f) where f is the
    // fraction of nonzero particles.
    fl::Real max_kl_divergence_;
};

}


#endif
