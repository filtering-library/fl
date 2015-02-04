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

#include <map>
#include <string>
#include <exception>

//#ifndef NDEBUG
    #if defined(__GNUC__)
        #define fl_current_function __PRETTY_FUNCTION__
    #elif defined(__FUNCSIG__)
        # define fl_current_function __FUNCSIG__
    #elif  defined(__func__)
        #define fl_current_function __func__
    #else
        #define fl_current_function "(unknown)"
    #endif

    #define fl_throw(excep) \
        fl::Exception::throw_exception(excep, \
                                       fl_current_function, \
                                       __FILE__, \
                                       __LINE__);
//#else
//    #define fl_throw(excep) throw excep;
//#endif

#if defined(__GNUC__)
    #define fl_attribute_noreturn __attribute__((noreturn))
#elif defined(_MSC_VER)
    #define fl_attribute_noreturn __declspec(noreturn)
#else
    #define fl_attribute_noreturn
#endif

namespace fl
{

/**
 * Genenric Exception
 *
 * \ingroup exceptions
 */
class Exception
        : public std::exception
{
public:
    /**
     * Create empty exception
     */
    Exception() = default;

    /**
     *
     */
    virtual ~Exception() throw() { }

    /**
     * Create an exception with a customized message under the tag \em Msg.
     *
     * @param message   Customized error message using the Msg tag.
     */
    explicit Exception(std::string msg)
    {
        info("Error", msg);
    }

    const char* what() const noexcept
    {
        std::string whats_wrong = "Dynamic exception "
                                  + name() + "\n" ;

        for (auto&& msg : diagnostic_)
        {
            whats_wrong += "[Exception " + name() + "] "
                            + msg.first + ": "
                            + msg.second + "\n";
        }
        //whats_wrong.pop_back(); not supported by C++0x
        whats_wrong.resize(whats_wrong.size () - 1);

        return whats_wrong.c_str();
    }


    /**
     * Sets a tagged diagnostic info message
     *
     * \param tag       Message tag
     * @param msg       Message text
     */
    void info(std::string tag, std::string msg)
    {
        diagnostic_[tag] = msg;
    }

    /**
     * \return Exception name
     */
    virtual std::string name() const noexcept { return "fl::Exception"; }

    /**
     * Throw an exception with diagnostic information
     */
    template <typename ExceptionType>
    fl_attribute_noreturn
    static inline void throw_exception(ExceptionType&& excep,
                                       const std::string& file,
                                       const std::string& function_name,
                                       int line)
    {
        excep.info("Function", function_name);
        excep.info("Line", std::to_string(line));
        excep.info("File", file);

        throw excep;
    }

protected:
    /**
     * \brief Diagnostic information. This may contain tagged messages used to
     * construct the what() message.
     */
    std::map<std::string, std::string> diagnostic_;
};

/**
 * \ingroup exceptions
 *
 * Index out of bounds exception
 */
class OutOfBoundsException
        : public Exception
{
public:    

    /**
     * Creates an OutOfBoundsException with a default message
     */
    OutOfBoundsException()
        : Exception("Index out of bounds")
    {
        // *this << OutOfBoundsInfo("Index out of bounds");
    }

    /**
     * Creates an OutOfBoundsException with a default message including the
     * index
     *
     * @param index     The out of bounds index
     */
    explicit OutOfBoundsException(long int index)
        : Exception("Index["
                    + std::to_string(index)
                    + "] out of bounds")
    { }

    /**
     * Creates an OutOfBoundsException with a default message including the
     * index and the container size
     *
     * @param index     The out of bounds index
     * @param size      Container size
     */
    OutOfBoundsException(long int index, long int size)
        : Exception("Index["
                    + std::to_string(index)
                    + "] out of bounds [0, "
                    + std::to_string(size)
                    + ")")
    { }

    /**
     * \return Exception name
     */
    virtual std::string name() const noexcept
    {
        return "fl::OutOfBoundsException";
    }
};

/**
 * \ingroup exceptions
 *
 * Exception representing a wrong size or dimension
 */
class WrongSizeException
        : public Exception
{
public:
    /**
     * Creates an WrongSizeException with a customized message
     */
    WrongSizeException(std::string msg)
        : Exception(msg)
    {
    }

    /**
     * Creates an WrongSizeException
     */
    WrongSizeException(size_t given, size_t expected)
        : Exception("Wrong size ("
                    + std::to_string(given)
                    + "). Expected ("
                    + std::to_string(expected)
                    + ")") { }

    /**
     * \return Exception name
     */
    virtual std::string name() const noexcept
    {
        return "fl::WrongSizeException";
    }
};

/**
 * \ingroup exceptions
 *
 * Exception representing an uninitialized dimension of a dynamic sized Gaussian
 */
class ZeroDimensionException
        : public Exception
{
public:
    /**
     * Creates an ZeroDimensionException
     */
    ZeroDimensionException(std::string entity = "Entity")
        : Exception(entity + " dimension is 0.") { }

    /**
     * \return Exception name
     */
    virtual std::string name() const noexcept
    {
        return "fl::ZeroDimensionException";
    }
};

/**
 * \ingroup exceptions
 *
 * Exception representing an attempt to resize a fixed-size entity
 */
class ResizingFixedSizeEntityException
        : public Exception
{
public:
    /**
     * Creates an ResizingFixedSizeEntityException
     */
    ResizingFixedSizeEntityException(size_t fixed_size,
                                     size_t new_size,
                                     std::string entity = "entity")
        : Exception("Attempt to resize the fixed-size ("
                    + std::to_string(fixed_size)
                    + ") "
                    + entity
                    + " to (" + std::to_string(new_size) + ")")
    { }

    /**
     * \return Exception name
     */
    virtual std::string name() const noexcept
    {
        return "fl::ResizingFixedSizeEntityException";
    }
};

}

#endif
