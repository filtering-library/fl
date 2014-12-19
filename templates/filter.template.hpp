/*
 * Copyright (c) <YEAR> <Author>
 *
 * <LICENSE>
 */

/**
 * \file FILTER_FILE_NAME.hpp
 * \date 
 * \author 
 */

#ifndef FL__FILTER__YOUR_FILTER_NAME_HPP
#define FL__FILTER__YOUR_FILTER_NAME_HPP

#include <fl/util/traits.hpp>
#include <fl/filter/filter_interface.hpp>
#include <fl/distribution/gaussian.hpp>

namespace fl
{
    
// Filter Forward declaration
class YOUR_FILTER_NAME;

// YOUR_FILTER_NAME Traits declaration
/**
 * \internal
 * \brief ...
 * 
 */
template <>
struct Traits< YOUR_FILTER_NAME >
{
    /*  Required  types */
    
    typedef std::shared_ptr< YOUR_FILTER_NAME > Ptr;
    
    /*
     * Define your State and Input if not provided somewhere
     * E.g.
     * 
     *   typedef Eigen::Matrix<double, 7, 1> State;
     *   typedef Eigen::Matrix<double, 6, 1> Input;
     *
     *  or if provided by the process model
     * 
     *   typedef typename Traits<ProcessModel>::State State;
     *   typedef typename Traits<ProcessModel>::Input Input; 
     * 
     * given a ProcessModel typedef in here
     */
    typedef YOUR_STATE_TYPE State;
    typedef YOUR_INPUT_TYPE Input;    
    
    /*
     * Define your Observation if not provided somewhere
     * E.g.
     * 
     *   typedef Eigen::Matrix<double, 13, 1> Observation;
     *
     *  or if provided by the process model
     * 
     *   typedef typename Traits<ObservationModel>::Observation Observation;
     * 
     * given a ObservationModel typedef in here
     */
    typedef YOUR_OBSERVATION_TYPE Observation;
    
    /*
     * Define the distribution type required by your filter.
     * Note that the distribution must implement the fl::Moments interface.
     * 
     * A common used distribution is the fl::Gaussian<State>.
     * 
     *   typedef ff::Gaussian<State> StateDistribution;
     */
    typedef YOUR_FILTER_DISTRIBUTION StateDistribution;
};


/**
 * \ingroup filter
 * 
 * \brief ...
 * 
 * \details ...
 */
template <>
class YOUR_FILTER_NAME
    : public FilterInterface< YOUR_FILTER_NAME >
{
public:    
    typedef YOUR_FILTER_NAME This;
    
    // make the Traits types available within this scope
    typedef typename Traits<This>::State             State;
    typedef typename Traits<This>::Input             Input;
    typedef typename Traits<This>::Observation       Observation;
    typedef typename Traits<This>::StateDistribution StateDistribution;
    
    
    // == implement the interface procedures ============================== // 
    
    virtual void predict(double delta_time,
                         const Input& input,
                         const StateDistribution& prior_dist,
                         StateDistribution& predicted_dist)
    {
        // here goes your filter prediction implementation
    }
       
    virtual void update(const Observation& observation,
                        const StateDistribution& predicted_dist,
                        StateDistribution& posterior_dist)
    {
        // here goes your filter update implementation
    }
        
    virtual void predict_and_update(double delta_time,
                                    const Input& input,
                                    const Observation& observation,
                                    const StateDistribution& prior_dist,
                                    StateDistribution& posterior_dist)
    {
        // this is the easiest solution
        predict(delta_time, input, prior_dist, posterior_dist);
        update(observation, posterior_dist, posterior_dist);
        
        // however, it might be far more efficient if you provide an 
        // implementation which exploits the all in-one-place structure
    }  
    
    // ...
};
}
          
#endif