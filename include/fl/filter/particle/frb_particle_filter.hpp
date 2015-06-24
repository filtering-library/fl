/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Max-Planck-Institute for Intelligent Systems,
 *                     University of Southern California
 *    Jan Issac (jan.issac@gmail.com)
 *    Manuel Wuthrich (manuel.wuthrich@gmail.com)
 *
 *
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

/**
 * @date 2015
 * @author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * Max-Planck-Institute for Intelligent Systems
 */


#ifndef FL__FILTER__PARTICLE__FRB_PARTICLE_FILTER_HPP
#define FL__FILTER__PARTICLE__FRB_PARTICLE_FILTER_HPP


#include <fl/distribution/discrete_distribution.hpp>
#include <fl/distribution/standard_gaussian.hpp>


/// \todo: REMOVE UNNECESSARY INCLUDES

#include <map>
#include <tuple>
#include <memory>

#include <fl/util/meta.hpp>
#include <fl/util/traits.hpp>
#include <fl/util/profiling.hpp>

#include <fl/exception/exception.hpp>
#include <fl/filter/filter_interface.hpp>


#include <fl/model/observation/joint_observation_model.hpp>

#include <Eigen/Core>

namespace fl
{

template <typename...> class FRBParticleFilter;

/**
 * FRBParticleFilter Traits
 */
template <typename StateTransitionFunction,
          typename SwitchingDensity,
          typename ObservationDensity>
struct Traits<FRBParticleFilter<StateTransitionFunction,
                                SwitchingDensity,
                                ObservationDensity>>
{

    /// \todo: does this make sense to just assume the state ans obsrv to be vectorxd?
    typedef typename Eigen::VectorXd State;
    typedef typename Eigen::VectorXd Obsrv;
    //    typedef typename StateTransitionFunction::State        State;
    //    typedef typename ObservationDensity::Obsrv             Obsrv;

    typedef typename StateTransitionFunction::Input        Input;

    typedef DiscreteDistribution<State> Belief;

    /** \cond INTERNAL */
    typedef typename StateTransitionFunction::Noise      ProcessNoise;
};

template <typename StateTransitionFunction,
          typename SwitchingDensity,
          typename ObservationDensity>
struct FRBParticleFilter<StateTransitionFunction,
                         SwitchingDensity,
                         ObservationDensity>
    :
    public FilterInterface<FRBParticleFilter<StateTransitionFunction,
                                             SwitchingDensity,
                                             ObservationDensity>>
{
private:
    typedef FRBParticleFilter<StateTransitionFunction,
                              SwitchingDensity,
                              ObservationDensity> This;

    typedef from_traits(ProcessNoise);

public:
    typedef from_traits(State);
    typedef from_traits(Input);
    typedef from_traits(Obsrv);
    typedef from_traits(Belief);

    typedef typename Eigen::Array<SwitchingDensity,
                                  Eigen::Dynamic, 1> SwitchingDensities;
    typedef typename Eigen::Array<ObservationDensity,
                                  Eigen::Dynamic, 2> ObservationDensities;

public:
    FRBParticleFilter(const StateTransitionFunction& trans_function,
                      const SwitchingDensities& switch_densities,
                      const ObservationDensities& obsrv_densities,
                      const double& max_kl_divergence = 1.0)
        : trans_function_(trans_function),
          switch_densities_(switch_densities),
          obsrv_densities_(obsrv_densities),
          process_noise_(trans_function.noise_dimension()),
          max_kl_divergence_(max_kl_divergence)
    { }


    /// predict ****************************************************************
    virtual void predict(double dt,
                         const Input& input,
                         const Belief& prior_belief,
                         Belief& predicted_belief)
    {
        predicted_belief = prior_belief;
        int global_dim = trans_function_.state_dimension();
        int local_dim = switch_densities_.size();


        // the global state part is propagated as in the standard particle filter
        for(size_t i = 0; i < predicted_belief.size(); i++)
        {
            Eigen::VectorXd& next_state =
                    predicted_belief.location(i).topRows(global_dim);

            Eigen::VectorXd& state = prior_belief.location(i).topRows(global_dim);

            next_state = trans_function_.predict_state(dt,
                                                       state,
                                                       process_noise_.sample(),
                                                       input);
        }

        // the local state part is propagated analytically
        for(size_t i = 0; i < predicted_belief.size(); i++)
        {
            Eigen::VectorXd& next_state =
                    predicted_belief.location(i).bottomRows(local_dim);

            Eigen::VectorXd& state =
                    prior_belief.location(i).bottomRows(local_dim);

            for(int j = 0; j < state.size(); j++)
            {
                next_state[j] =
                     switch_densities_[j].probability(1, 1, dt) * state[j]
                   + switch_densities_[j].probability(1, 0, dt) * (1.-state[j]);
            }
        }
    }

    /// update *****************************************************************
    virtual void update(const Obsrv& obsrv,
                        const Belief& predicted_belief,
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

        int global_dim = trans_function_.state_dimension();
        int local_dim = switch_densities_.size();

        auto delta_log_prob =
                Belief::Function::Zero(predicted_belief.size());

        for(int s = 0; s < predicted_belief.size(); s++)
        {
            Eigen::VectorXd& global_state =
                    predicted_belief.location(s).topRows(global_dim);

            Eigen::VectorXd& local_state =
                    predicted_belief.location(s).bottomRows(local_dim);

            for(int o = 0; o < obsrv.size(); o++)
            {
                Real p_given1 =
                        obsrv_densities_[o,1].probability(obsrv[o], global_state);

                Real p_given0 =
                        obsrv_densities_[o,0].probability(obsrv[o], global_state);

                Real p =   p_given0 * (1. - local_state[o])
                                  + p_given1 * local_state[o];

            // there is a problem if we want to use this filter for tracking:
            // in tracking we only update the stuff which needs to be updated,
            // not all sensors are updated
            }


        }


        // update the weights of the particles with the likelihoods
        posterior_belief.delta_log_prob_mass(
             obsrv_density_.log_probabilities(obsrv, predicted_belief.locations()));
    }

    /// predict and update *****************************************************
    virtual void predict_and_update(double dt,
                                    const Input& input,
                                    const Obsrv& observation,
                                    const Belief& prior_belief,
                                    Belief& posterior_belief)
    {
        predict(dt, input, prior_belief, posterior_belief);
        update(observation, posterior_belief, posterior_belief);
    }


    /// set and get ************************************************************
    StateTransitionFunction& trans_function()
    {
        return trans_function_;
    }
    ObservationDensity& obsrv_density()
    {
        return obsrv_density_;
    }

    const StateTransitionFunction& trans_function() const
    {
        return trans_function_;
    }

    const ObservationDensity& obsrv_density() const
    {
        return obsrv_density_;
    }

    /// \todo: should this function be here?
    virtual Belief create_state_distribution() const
    {
        auto state_distr = Belief(trans_function().state_dimension());

        return state_distr;
    }

protected:
    StateTransitionFunction trans_function_;
    ObservationDensity obsrv_density_;

    SwitchingDensities switch_densities_;
    ObservationDensities obsrv_densities_;


    StandardGaussian<ProcessNoise> process_noise_;

    // when the KL divergence KL(p||u), where p is the particle distribution
    // and u is the uniform distribution, exceeds max_kl_divergence_, then there
    // is a resampling step. can be understood as -log(f) where f is the
    // fraction of nonzero particles.
    double max_kl_divergence_;
};

}


#endif
