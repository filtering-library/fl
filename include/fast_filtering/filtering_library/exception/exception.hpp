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
 * @date 10/21/2014
 * @author Jan Issac (jan.issac@gmail.com)
 * @author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * Max-Planck-Institute for Intelligent Systems,
 * University of Southern California
 */

#ifndef FL__FILTER__EXCEPTION__EXCEPTION_HPP
#define FL__FILTER__EXCEPTION__EXCEPTION_HPP

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

}

#endif
