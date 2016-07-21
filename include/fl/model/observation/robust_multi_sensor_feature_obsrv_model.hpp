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
 * \file robust_multi_sensor_feature_obsrv_model.hpp
 * \date September 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

#include <fl/util/meta.hpp>
#include <fl/util/traits.hpp>
#include <fl/util/types.hpp>
#include <fl/util/descriptor.hpp>
#include <fl/distribution/gaussian.hpp>
#include <fl/model/observation/robust_feature_obsrv_model.hpp>

namespace fl
{

/**
 * \ingroup observation_models
 *
 * \brief Represents an observation function or model which takes an arbitrary
 *        observation model (one that implements an SensorFunction & a
 *        an SensorDensity) as an argument and maps it into a feature space
 *        used by the RobustGaussianFilter.
 */
template <typename Sensor, int SensorsCount>
class MultiRobustSensorFunction
    : public RobustSensorFunction<Sensor>

{
private:
    typedef RobustSensorFunction<Sensor> RobustSensorFunctionBase;

public:
    /**
     * \brief \a InputObsrv type which is the same as
     *        Sensor::Obsrv. \a InputObsrv is mapped into the feature space.
     *        The resulting type is \a Obsrv.
     */
    typedef typename Sensor::Obsrv InputObsrv;

public:
    // Remove body_moments(mean, cov) of RobustSensorFunction from public
    // interface.
    using RobustSensorFunctionBase::body_moments;

public:
    /**
     * \brief Constructs a robust feature observation model for the robust
     *        gaussian filter
     *
     * \param obsrv_model
     *          Reference to the source observation model
     *
     * \note This model takes only a reference of to an existing lvalue of the
     *       source model
     */
    explicit MultiRobustSensorFunction(
            Sensor& obsrv_model,
            int sensor_count)
        : RobustSensorFunctionBase(obsrv_model),
          body_gaussians_(sensor_count),
          id_(0)
    { }

    /**
     * \brief Overridable default destructor
     */
    virtual ~MultiRobustSensorFunction() noexcept { }

    virtual std::string name() const
    {
        return "MultiRobustSensorFunction<"
                + this->list_arguments(
                      this->embedded_obsrv_model().name())
                + ">";
    }

    virtual std::string description() const
    {
        return "Robust feature observation model for multi-sensor filtering with "
                + this->list_descriptions(
                      this->embedded_obsrv_model().description());
    }

    /**
     * \brief Sets the feature function body moments for the specified sensor
     *        with the ID \ id
     * \param body_gaussian
     *          \f${\cal N}(y_t\mid \mu_{y}, \Sigma_{yy})\f$
     *
     * \todo PAPER REF
     */
    virtual void body_moments(
        const typename FirstMomentOf<InputObsrv>::Type& mean_obsrv,
        const typename SecondMomentOf<InputObsrv>::Type& cov_obsrv,
        int id)
    {
        assert(mean_obsrv.size() == cov_obsrv.rows());
        assert(mean_obsrv.size() == cov_obsrv.cols());

        body_gaussians_(id).dimension(mean_obsrv.size());
        body_gaussians_(id).mean(mean_obsrv);
        body_gaussians_(id).covariance(cov_obsrv);
    }

    /**
     * \return Model id number
     *
     * In case of multiple sensors of the same kind, this function returns the
     * id of the individual model.
     */
    int id() const
    {
        return id_;
    }

    /**
     * \brief Sets the model id for mulit-sensor use
     *
     * In some cases a single observation model may be used for multiple sensor
     * of the same kind. It often suffices to alter the sensor id before
     * evaluating the model.
     *
     * \param new_id    Model's new ID
     *
     * Sets the current feature model id and updates the body_gaussian for this
     * particular sensor
     */
    void id(int new_id) override
    {
        id_ = new_id;

        RobustSensorFunctionBase::id(new_id);

        RobustSensorFunctionBase::body_moments(
            body_gaussians_(id_).mean(),
            body_gaussians_(id_).covariance());
    }

    Eigen::Array<Gaussian<InputObsrv>, SensorsCount, 1>& body_gaussians()
    {
        return body_gaussians_;
    }

    Gaussian<InputObsrv>& body_gaussian()
    {
        return body_gaussians_(id_);
    }

protected:
    /** \cond internal */
    Eigen::Array<Gaussian<InputObsrv>, SensorsCount, 1> body_gaussians_;
    int id_;
    /** \endcond */
};

}
