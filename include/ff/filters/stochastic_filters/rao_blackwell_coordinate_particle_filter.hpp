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
 * \file rao_blackwell_coordinate_particle_filter.hpp
 * \date 05/25/2014
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */


#ifndef FAST_FILTERING_FILTERS_STOCHASTIC_RAO_BLACKWELL_COORDINATE_PARTICLE_FILTER_HPP
#define FAST_FILTERING_FILTERS_STOCHASTIC_RAO_BLACKWELL_COORDINATE_PARTICLE_FILTER_HPP

#include <vector>
#include <limits>
#include <string>
#include <algorithm>
#include <cmath>
#include <memory>

#include <Eigen/Core>

#include <fl/util/math.hpp>
#include <fl/util/traits.hpp>
#include <fl/util/profiling.hpp>
#include <fl/util/assertions.hpp>
#include <fl/util/discrete_distribution.hpp>

#include <fl/distribution/gaussian.hpp>
#include <fl/distribution/sum_of_deltas.hpp>
#include <fl/distribution/interface/standard_gaussian_mapping.hpp>
#include <fl/model/process/process_model_interface.hpp>

#include <ff/models/observation_models/interfaces/rao_blackwell_observation_model.hpp>

namespace fl
{

/// \todo MISSING DOC. MISSING UTESTS

template<typename ProcessModel, typename ObservationModel>
class RaoBlackwellCoordinateParticleFilter
{
public:
    typedef typename Traits<ProcessModel>::Scalar Scalar;
    typedef typename Traits<ProcessModel>::State  State;
    typedef typename Traits<ProcessModel>::Input  Input;
    typedef typename Traits<ProcessModel>::Noise  Noise;

    typedef typename ObservationModel::Observation Observation;

    // state distribution
    typedef SumOfDeltas<State> StateDistributionType;

public:
    RaoBlackwellCoordinateParticleFilter(
            const std::shared_ptr<ProcessModel> process_model,
            const std::shared_ptr<ObservationModel>  observation_model,
            const std::vector<std::vector<size_t>>& sampling_blocks,
            const Scalar& max_kl_divergence = 0):
        observation_model_(observation_model),
        process_model_(process_model),
        max_kl_divergence_(max_kl_divergence)
    {
        static_assert_base(
            ProcessModel,
            ProcessModelInterface<State, Noise, Input>);

        static_assert_base(
            ProcessModel,
            StandardGaussianMapping<State, Noise>);

        static_assert_base(
            ObservationModel,
            RaoBlackwellObservationModel<State, Observation>);

        SamplingBlocks(sampling_blocks);
    }

    virtual ~RaoBlackwellCoordinateParticleFilter() { }

public:
    void Filter(const Observation&  observation,
                const Scalar&       delta_time,
                const Input&        input)
    {
        observation_model_->SetObservation(observation, delta_time);

        loglikes_ = std::vector<Scalar>(samples_.size(), 0);
        noises_ = std::vector<Noise>(samples_.size(), Noise::Zero(process_model_->noise_dimension()));
        next_samples_ = samples_;

        for(size_t block_index = 0; block_index < sampling_blocks_.size(); block_index++)
        {
            for(size_t particle_index = 0; particle_index < samples_.size(); particle_index++)
            {
                for(size_t i = 0; i < sampling_blocks_[block_index].size(); i++)
                    noises_[particle_index](sampling_blocks_[block_index][i]) = unit_gaussian_.sample()(0);

                next_samples_[particle_index] =
                        process_model_->predict_state(delta_time,
                                                      samples_[particle_index],
                                                      noises_[particle_index],
                                                      input);
            }

            bool update_occlusions = (block_index == sampling_blocks_.size()-1);
            std::vector<Scalar> new_loglikes = observation_model_->Loglikes(next_samples_,
                                                                           indices_,
                                                                           update_occlusions);
            std::vector<Scalar> delta_loglikes(new_loglikes.size());
            for(size_t i = 0; i < delta_loglikes.size(); i++)
                delta_loglikes[i] = new_loglikes[i] - loglikes_[i];
            loglikes_ = new_loglikes;
            UpdateWeights(delta_loglikes);
        }

        samples_ = next_samples_;
        state_distribution_.SetDeltas(samples_); // not sure whether this is the right place
    }

    void Resample(const size_t& sample_count)
    {
        std::vector<State> samples(sample_count);
        std::vector<size_t> indices(sample_count);
        std::vector<Noise> noises(sample_count);
        std::vector<State> next_samples(sample_count);
        std::vector<Scalar> loglikes(sample_count);

        hf::DiscreteDistribution sampler(log_weights_);

        for(size_t i = 0; i < sample_count; i++)
        {
            size_t index = sampler.sample();

            samples[i]      = samples_[index];
            indices[i]      = indices_[index];
            noises[i]       = noises_[index];
            next_samples[i] = next_samples_[index];
            loglikes[i]     = loglikes_[index];
        }
        samples_        = samples;
        indices_        = indices;
        noises_         = noises;
        next_samples_   = next_samples;
        loglikes_       = loglikes;

        log_weights_        = std::vector<Scalar>(samples_.size(), 0.);

        state_distribution_.SetDeltas(samples_); // not sure whether this is the right place
    }

private:
    // O(6n + nlog(n)) might be reducible to O(4n)
    void UpdateWeights(std::vector<Scalar> log_weight_diffs)
    {
        for(size_t i = 0; i < log_weight_diffs.size(); i++)
            log_weights_[i] += log_weight_diffs[i];

        std::vector<Scalar> weights = log_weights_;
        // descendant sorting
        std::sort(weights.begin(), weights.end(), std::greater<Scalar>());

        for(int i = weights.size() - 1; i >= 0; i--)
            weights[i] -= weights[0];

        std::for_each(weights.begin(), weights.end(), [](Scalar& w){ w = std::exp(w); });

        weights = fl::normalize(weights, Scalar(1));

        // compute KL divergence to uniform distribution KL(p|u)
        Scalar kl_divergence = std::log(Scalar(weights.size()));
        for(size_t i = 0; i < weights.size(); i++)
        {
            Scalar information = - std::log(weights[i]) * weights[i];
            if(!std::isfinite(information))
                information = 0; // the limit for weight -> 0 is equal to 0
            kl_divergence -= information;
        }

        if(kl_divergence > max_kl_divergence_)
            Resample(samples_.size());
    }

public:
    // set
    void Samples(const std::vector<State >& samples)
    {
        samples_ = samples;
        indices_ = std::vector<size_t>(samples_.size(), 0); observation_model_->Reset();
        log_weights_ = std::vector<Scalar>(samples_.size(), 0);
    }
    void SamplingBlocks(const std::vector<std::vector<size_t>>& sampling_blocks)
    {
        sampling_blocks_ = sampling_blocks;

        // make sure sizes are consistent
        size_t dimension = 0;
        for(size_t i = 0; i < sampling_blocks_.size(); i++)
            for(size_t j = 0; j < sampling_blocks_[i].size(); j++)
                dimension++;

        if(dimension != process_model_->standard_variate_dimension())
        {
            std::cout << "the dimension of the sampling blocks is " << dimension
                      << " while the dimension of the noise is "
                      << process_model_->standard_variate_dimension() << std::endl;
            exit(-1);
        }
    }

    // get
    const std::vector<State>& Samples() const
    {
        return samples_;
    }

    StateDistributionType& StateDistribution()
    {
        return state_distribution_;
    }

private:
    // internal state TODO: THIS COULD BE MADE MORE COMPACT!!
    StateDistributionType state_distribution_;

    std::vector<State > samples_;
    std::vector<size_t> indices_;
    std::vector<Scalar>  log_weights_;
    std::vector<Noise> noises_;
    std::vector<State> next_samples_;
    std::vector<Scalar> loglikes_;

    // observation model
    std::shared_ptr<ObservationModel> observation_model_;

    // process model
    std::shared_ptr<ProcessModel> process_model_;

    // parameters
    std::vector<std::vector<size_t>> sampling_blocks_;
    Scalar max_kl_divergence_;

    // distribution for sampling
    Gaussian<Eigen::Matrix<Scalar,1,1>> unit_gaussian_;
};

}

#endif
