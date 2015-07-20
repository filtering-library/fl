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

class Descriptor
{
public:
    virtual std::string name() const = 0;

    virtual std::string description() const = 0;

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

    template <typename ... Elements>
    std::string list_arguments(const Elements& ... elements) const
    {
        std::tuple<const Elements&...> element_tuple{elements...};

        std::string list_string;
        list_impl<sizeof...(Elements), 0>(
            element_tuple, "", ",", ",", list_string);

        return list_string;
    }

    template <typename ... Elements>
    std::string list_descriptions(const Elements& ... elements) const
    {
        std::tuple<const Elements&...> element_tuple{elements...};

        std::string list_string;
        list_impl<sizeof...(Elements), 0>(
            element_tuple, "- ", ",", ", and", list_string);

        return list_string;
    }

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
