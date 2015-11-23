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
 * \file descriptor_test.hpp
 * \date July 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>

#include <fl/util/descriptor.hpp>

template <typename...> class Dummy;

template <>
class Dummy<>
    : public fl::Descriptor
{
public:
    virtual std::string name() const
    {
        return "Dummy<>";
    }

    virtual std::string description() const
    {
        return "Dummy class implementing the fl::Descriptor";
    }
};

template <typename D>
class NestedDummy
    : public D
{
public:
    using D::indent;

    virtual std::string name() const
    {
        return "NestedDummy<" + indent(D::name()) + ">";
    }

    virtual std::string description() const
    {
        return "NestedDummy class implementing the fl::Descriptor with"
                + indent(D::description());
    }
};

TEST(DescriptorTests, Dummy_name)
{
    Dummy<> dummy;
    EXPECT_STREQ(dummy.name().c_str(), "Dummy<>");
}

TEST(DescriptorTests, Dummy_description)
{
    Dummy<> dummy;
    EXPECT_STREQ(dummy.description().c_str(),
                 "Dummy class implementing the fl::Descriptor");
}

TEST(DescriptorTests, Dummy_indent)
{
    Dummy<> dummy;

    EXPECT_STREQ(dummy.indent(dummy.description()).c_str()  ,
                 "\n    Dummy class implementing the fl::Descriptor");
}

TEST(DescriptorTests, NestedDummy_name)
{
    NestedDummy<Dummy<>> dummy;
    EXPECT_STREQ(dummy.name().c_str(), "NestedDummy<\n    Dummy<>>");
}

TEST(DescriptorTests, NestedDummy_description)
{
    NestedDummy<Dummy<>> dummy;
    EXPECT_STREQ(dummy.description().c_str(),
                 "NestedDummy class implementing the fl::Descriptor with"
                 "\n    Dummy class implementing the fl::Descriptor");
}

TEST(DescriptorTests, NestedDummy_indent)
{
    NestedDummy<Dummy<>> dummy;

    EXPECT_STREQ(dummy.indent(dummy.description()).c_str()  ,
                 "\n    NestedDummy class implementing the fl::Descriptor with"
                 "\n        Dummy class implementing the fl::Descriptor");
}


TEST(DescriptorTests, NestedNestedDummy_name)
{
    NestedDummy<NestedDummy<Dummy<>>> dummy;

    EXPECT_STREQ(dummy.name().c_str(),
                 "NestedDummy<\n    NestedDummy<\n        Dummy<>>>");
}

TEST(DescriptorTests, NestedNestedDummy_description)
{
    NestedDummy<NestedDummy<Dummy<>>> dummy;

    EXPECT_STREQ(dummy.description().c_str(),
                 "NestedDummy class implementing the fl::Descriptor with"
                 "\n    NestedDummy class implementing the fl::Descriptor with"
                 "\n        Dummy class implementing the fl::Descriptor");
}
