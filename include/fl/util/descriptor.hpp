/*
 * This is part of the FL library, a C++ Bayesian filtering library
 * (https://github.com/filtering-library)
 *
 * Copyright (c) 2014 Jan Issac (jan.issac@gmail.com)
 * Copyright (c) 2014 Manuel Wuthrich (manuel.wuthrich@gmail.com)
 *
 * This Source Code Form is subject to the terms of the MIT License (MIT).
 * A copy of the license can be found in the LICENSE file distributed with this
 * source code.
 */

/**
 * \file descriptor.hpp
 * \date July 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#ifndef FL__UTIL__DESCRIPTOR_HPP
#define FL__UTIL__DESCRIPTOR_HPP

#include <string>
#include <tuple>

namespace fl
{

/**
 * \ingroup util
 *
 * Descriptor represents the description of a construct. In various cases a
 * class is composed on compile-time. It is often hard to know what the final
 * composition constituents are after compilation. The Descriptor provides
 * the name of composed class and a full description of it.
 */
class Descriptor
{
public:
    /**
     * \brief Generates the name of this construct (compile time generated
     *        class). This is used to determine the name of the class.
     *
     * This function returns the name of this composition including its
     * constitutent parts. The resulting name is often of a template class
     * definition form, e.g. in case of a Kalman filter
     *
     * \code
     * GaussianFilter<
     *     LinearStateTransitionModel<State, Noise, Input>,
     *     LinearGaussianObservationModel<Obsrv, State, Noise>>
     * \endcode
     *
     * \return Name of \c *this construct
     */
    virtual std::string name() const = 0;

    /**
     * \brief Generates of the description this construct (compile time
     *        generated class). This is used to determine the name of the class.
     *
     * This function generates the full description of this class which is
     * composed on compile-time. The description typically contains the
     * descriptions of all constituent parts of this class.
     *
     * \return Full description of \c *this construct
     */
    virtual std::string description() const = 0;

protected:
    /** \cond internal */

    /**
     * \brief indent a given text
     *
     * \param [in] text     Text to indent
     * \param [in] levels   Levels of indentation
     *
     * \return indented text by the number of level
     */
    virtual std::string indent(const std::string& text, int levels = 1) const
    {
        std::istringstream iss(text);
        std::string indented_text;
        std::string line;
        while(std::getline(iss, line))
        {
            indented_text += "\n" + std::string(4 * levels, ' ') + line;
        }

        return indented_text;
    }

    /**
     * Creates list with the correct indentation out of the passed elements.
     */
    template <typename ... Elements>
    std::string list_arguments(const Elements& ... elements) const
    {
        std::tuple<const Elements&...> element_tuple{elements...};

        std::string list_string;
        list_impl<sizeof...(Elements), 0>(
            element_tuple, "", ",", ",", list_string);

        return list_string;
    }

    /**
     * Creates list of descriptions with the correct indentation
     */
    template <typename ... Elements>
    std::string list_descriptions(const Elements& ... elements) const
    {
        std::tuple<const Elements&...> element_tuple{elements...};

        std::string list_string;
        list_impl<sizeof...(Elements), 0>(
            element_tuple, "- ", ",", ", and", list_string);

        return list_string;
    }
    /** \endcond */

private:
    template <int Size, int k, typename ... Elements>
    void list_impl(const std::tuple<const Elements&...>& element_tuple,
                   const std::string& bullet,
                   const std::string& delimiter,
                   const std::string& final_delimiter,
                   std::string& list_string) const
    {
        auto&& element = std::get<k>(element_tuple);

        list_string += indent(bullet + element);

        if (k + 2 < Size)
        {
            list_string += delimiter;
        }
        else if (k + 1 < Size)
        {
            list_string += final_delimiter;
        }

        if (Size == k + 1) return;

        list_impl<Size, k + (k + 1 < Size ? 1 : 0)>(
            element_tuple,
            bullet,
            delimiter,
            final_delimiter,
            list_string);
    }
};

}

#endif
