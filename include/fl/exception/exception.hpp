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
 * \file exception.hpp
 * \date October 2014
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__EXCEPTION__EXCEPTION_HPP
#define FL__EXCEPTION__EXCEPTION_HPP

#include <string>
#include <exception>
#include <boost/exception/all.hpp>
#include <boost/lexical_cast.hpp>

namespace fl
{

/**
 * Genenric Exception
 *
 * \ingroup exceptions
 */
class Exception:
        virtual public std::exception,
        virtual public boost::exception
{
public:
    /**
     * Create empty exception
     */
    Exception() = default;

    /**
     * Create an exception with a customized message under the tag \em Msg.
     *
     * @param message   Customized error message using the Msg tag.
     */
    explicit Exception(std::string message)
    {
        *this << boost::error_info<struct Msg, std::string>(message);
    }

    const char* what() const noexcept
    {
        return boost::diagnostic_information_what(*this);
    }
};

/**
 * \ingroup exceptions
 *
 * Index out of bounds exception
 */
class OutOfBoundsException:
        public Exception
{
public:
    typedef boost::error_info<struct OutOfBounds, std::string> OutOfBoundsInfo;

    /**
     * Creates an OutOfBoundsException with a default message
     */
    OutOfBoundsException()
    {
        *this << OutOfBoundsInfo("Index out of bounds");
    }

    /**
     * Creates an OutOfBoundsException with a default message including the
     * index
     *
     * @param index     The out of bounds index
     */
    explicit OutOfBoundsException(long int index)
    {
        *this << OutOfBoundsInfo(
                    "Index("
                    + boost::lexical_cast<std::string>(index)
                    + ") out of bounds");
    }

    /**
     * Creates an OutOfBoundsException with a default message including the
     * index and the container size
     *
     * @param index     The out of bounds index
     * @param size      Container size
     */
    OutOfBoundsException(long int index, long int size)
    {
        *this << OutOfBoundsInfo(
                    "Index["
                    + boost::lexical_cast<std::string>(index)
                    + "] out of bounds [0, "
                    + boost::lexical_cast<std::string>(size)
                    + ")");
    }
};

/**
 * \ingroup exceptions
 *
 * Exception representing a wrong size or dimension
 */
class WrongSizeException:
    public fl::Exception
{
public:
    /**
     * Creates an WrongSizeException with a customized message
     */
    WrongSizeException(std::string msg):
        fl::Exception(msg)
    {
    }

    /**
     * Creates an WrongSizeException
     */
    WrongSizeException(size_t given, size_t expected):
        fl::Exception(
            "Wrong size ("
            + boost::lexical_cast<std::string>(given)
            + "). Expected ("
            + boost::lexical_cast<std::string>(expected)
            + ")") { }
};

/**
 * \ingroup exceptions
 *
 * Exception representing an uninitialized dimension of a dynamic sized Gaussian
 */
class ZeroDimensionException:
    public fl::Exception
{
public:
    /**
     * Creates an ZeroDimensionException
     */
    ZeroDimensionException(std::string entity = "Entity"):
        fl::Exception(entity + " dimension is 0.") { }
};

/**
 * \ingroup exceptions
 *
 * Exception representing an attempt to resize a fixed-size entity
 */
class ResizingFixedSizeEntityException:
    public fl::Exception
{
public:
    /**
     * Creates an ResizingFixedSizeEntityException
     */
    ResizingFixedSizeEntityException(size_t fixed_size,
                                     size_t new_size,
                                     std::string entity = "entity"):
        fl::Exception("Attempt to resize the fixed-size ("
                      + boost::lexical_cast<std::string>(fixed_size)
                      + ") "
                      + entity
                      + " to ("
                      + boost::lexical_cast<std::string>(new_size)
                      + ")") { }
};

}

#endif
