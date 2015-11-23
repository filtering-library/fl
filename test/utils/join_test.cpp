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
 * \file options_test.hpp
 * \date Febuary 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <gtest/gtest.h>

#include <type_traits>

#include <fl/util/meta.hpp>

using namespace fl;

template <int i> struct O { };
template <int i> struct P { };

template <typename ... O> struct JO;
template <typename ... P> struct JP;

TEST(MetaJoinTests, SimpleJoin)
{
    typedef CollapseJoin<
                JO<>,
                TypeSequence<>,
                Join<O<0>, O<1>, O<2>>
            >::Type ResultType;

    EXPECT_TRUE((std::is_same<
                    ResultType,
                    JO<O<0>, O<1>, O<2>>
                 >::value));
}

TEST(MetaJoinTests, NestedJoin)
{
    typedef CollapseJoin<
                JO<>,
                TypeSequence<>,
                Join<Join<O<0>, O<1>>>
            >::Type ResultType;

    EXPECT_TRUE((std::is_same<
                    ResultType,
                    JO<O<0>, O<1>>
                 >::value));

    EXPECT_FALSE((std::is_same<
                    ResultType,
                    JO<O<1>, O<0>>
                 >::value));
}

TEST(MetaJoinTests, DoubleNestedJoin)
{
    typedef CollapseJoin<
                JO<>,
                TypeSequence<>,
                Join<O<0>, O<1>>
            >::Type ResultType1;

    typedef CollapseJoin<
                JO<>,
                TypeSequence<>,
                Join<Join<O<0>, O<1>>>
            >::Type ResultType2;

    typedef CollapseJoin<
                JO<>,
                TypeSequence<>,
                Join<Join<Join<O<0>, O<1>>>>
            >::Type ResultType3;

    EXPECT_TRUE((std::is_same<ResultType1, ResultType2>::value));
    EXPECT_TRUE((std::is_same<ResultType2, ResultType3>::value));
    EXPECT_TRUE((std::is_same<ResultType3, JO<O<0>, O<1>>>::value));
}

TEST(MetaJoinTests, Associativity)
{
    typedef CollapseJoin<
                JO<>,
                TypeSequence<>,
                Join<Join<O<0>, O<1>>, O<2>>
            >::Type ResultType1;

    typedef CollapseJoin<
                JO<>,
                TypeSequence<>,
                Join<O<0>, Join<O<1>, O<2>>>
            >::Type ResultType2;

    EXPECT_TRUE((std::is_same<ResultType1, ResultType2>::value));
    EXPECT_TRUE((std::is_same<ResultType2, JO<O<0>, O<1>, O<2>>>::value));
}

/*
 * doesn"t compile as expected:
 * error: static assertion failed: "Join<A, B, ...> operator must take 2 or more
 * operands"
 */
//TEST(MetaJoinTests, SingleArgument_should_not_compile)
//{
//    typedef CollapseJoin<
//                JO<>,
//                TypeSequence<>,
//                Join<O<0>>
//            >::Type ResultType;
//}

TEST(MetaJoinTests, ExpandStatic)
{
    typedef CollapseJoin<
                JO<>,
                TypeSequence<>,
                Join<MultipleOf<O<0>, 3>>
            >::Type ResultType;

    EXPECT_TRUE((std::is_same<
                    ResultType,
                    JO<MultipleOf<O<0>, 3>>
                 >::value));
}

TEST(MetaJoinTests, ExpandDynamic)
{
    typedef CollapseJoin<
                JO<>,
                TypeSequence<>,
                Join<MultipleOf<O<0>, -1>>
            >::Type ResultType;

    EXPECT_TRUE((std::is_same<
                    ResultType,
                    JO<MultipleOf<O<0>, -1>>
                 >::value));

    EXPECT_FALSE((std::is_same<
                    ResultType,
                    JO<MultipleOf<O<1>, -1>>
                 >::value));
}

TEST(MetaJoinTests, NextedExpandStatic)
{
    typedef CollapseJoin<
                JO<>,
                TypeSequence<>,
                Join<Join<MultipleOf<O<0>, 3>>>
            >::Type ResultType;

    EXPECT_TRUE((std::is_same<
                    ResultType,
                    JO<MultipleOf<O<0>, 3>>
                 >::value));
}

TEST(MetaJoinTests, Join_Ox_Expand)
{
    typedef CollapseJoin<
                JO<>,
                TypeSequence<>,
                Join<O<0>, MultipleOf<O<1>, 3>>
            >::Type ResultType;

    EXPECT_TRUE((std::is_same<
                    ResultType,
                    JO<O<0>, JO<MultipleOf<O<1>, 3>>>
                 >::value));
}


TEST(MetaJoinTests, Join_Expand_Ox)
{
    typedef CollapseJoin<
                JO<>,
                TypeSequence<>,
                Join<MultipleOf<O<1>, 3>, O<0>>
            >::Type ResultType;

    EXPECT_TRUE((std::is_same<
                    ResultType,
                    JO<JO<MultipleOf<O<1>, 3>>, O<0>>
                 >::value));
}

TEST(MetaJoinTests, Join_JoinExpand_Ox)
{
    typedef CollapseJoin<
                JO<>,
                TypeSequence<>,
                Join<Join<MultipleOf<O<1>, 3>>, O<0>>
            >::Type ResultType;

    EXPECT_TRUE((std::is_same<
                    ResultType,
                    JO<JO<MultipleOf<O<1>, 3>>, O<0>>
                 >::value));
}


TEST(MetaJoinTests, JoinAssociativityWithExpand)
{
    typedef CollapseJoin<
                JO<>,
                TypeSequence<>,
                Join<Join<Join<MultipleOf<O<0>, 3>>, O<1>>, O<2>>
            >::Type ResultType1;

    typedef CollapseJoin<
                JO<>,
                TypeSequence<>,
                Join<Join<MultipleOf<O<0>, 3>>, Join<O<1>>, O<2>>
            >::Type ResultType2;

    EXPECT_TRUE((std::is_same<ResultType1, ResultType2>::value));

    EXPECT_TRUE((std::is_same<
                    ResultType2,
                    JO<JO<MultipleOf<O<0>, 3>>, O<1>, O<2>>
                >::value));
}

//TEST(MetaJoinTests, JoinAdaptive_Ox)
//{
//    typedef CollapseJoin<
//                JO<>,
//                TypeSequence<>,
//                Join<Adaptive<O<0>, P<0>>, O<1>>
//            >::Type ResultType;

//    EXPECT_TRUE((std::is_same<
//                    ResultType,
//                    Adaptive<
//                        JO<O<0>, NotAdaptive<O<1>>>,
//                    JointProcessModel<P<0>, NAProcessModel>>
//                 >::value));
//}

//TEST(MetaJoinTests, Join_Ox_Adaptive)
//{
//    typedef CollapseJoin<
//                JO<>,
//                TypeSequence<>,
//                Join<O<0>, Adaptive<O<1>, P<1>>>
//            >::Type ResultType;

//    EXPECT_TRUE((std::is_same<
//                    ResultType,
//                    Adaptive<
//                        JO<
//                            NotAdaptive<O<0>>, O<1>>,
//                            JointProcessModel<NAProcessModel, P<1>>
//                    >
//                 >::value));
//}

//TEST(MetaJoinTests, Join_Ox_Adaptive_Ox)
//{
//    typedef CollapseJoin<
//                JO<>,
//                TypeSequence<>,
//                Join<O<0>, Adaptive<O<1>, P<1>>, O<2>>
//            >::Type ResultType;

//    EXPECT_TRUE((std::is_same<
//                    ResultType,
//                    Adaptive<
//                        JO<NotAdaptive<O<0>>, O<1>, NotAdaptive<O<2>>>,
//                        JointProcessModel<
//                            NAProcessModel,
//                            P<1>,
//                            NAProcessModel>
//                        >
//                 >::value));
//}


//TEST(MetaJoinTests, Join_Ox_Ox_Adaptive)
//{
//    typedef CollapseJoin<
//                JO<>,
//                TypeSequence<>,
//                Join<O<0>, O<1>, Adaptive<O<2>, P<2>>>
//            >::Type ResultType;

//    EXPECT_TRUE((std::is_same<
//                    ResultType,
//                    Adaptive<
//                        JO<NotAdaptive<O<0>>, NotAdaptive<O<1>>, O<2>>,
//                        JointProcessModel<
//                            NAProcessModel,
//                            NAProcessModel, P<2>>
//                        >
//                 >::value));
//}

//TEST(MetaJoinTests, Join_Adaptive_Ox_Ox)
//{
//    typedef CollapseJoin<
//                JO<>,
//                TypeSequence<>,
//                Join<Adaptive<O<0>, P<0>>, O<1>, O<2>>
//            >::Type ResultType;

//    EXPECT_TRUE((std::is_same<
//                    ResultType,
//                    Adaptive<
//                        JO<O<0>, NotAdaptive<O<1>>, NotAdaptive<O<2>>>,
//                        JointProcessModel<
//                            P<0>,
//                            NAProcessModel,
//                            NAProcessModel>>
//                 >::value));
//}

//TEST(MetaJoinTests, Join_AllAdaptive)
//{
//    typedef CollapseJoin<
//                JO<>,
//                TypeSequence<>,
//                Join<
//                    Adaptive<O<0>, P<0>>,
//                    Adaptive<O<1>, P<1>>,
//                    Adaptive<O<2>, P<2>>
//                >
//            >::Type ResultType;

//    EXPECT_TRUE((std::is_same<
//                    ResultType,
//                    Adaptive<
//                        JO<O<0>, O<1>, O<2>>,
//                        JointProcessModel<P<0>, P<1>, P<2>>
//                    >
//                 >::value));
//}

//TEST(MetaJoinTests, JoinExpandAdaptive)
//{
//    typedef CollapseJoin<
//                JO<>,
//                TypeSequence<>,
//                Join<MultipleOf<Adaptive<O<0>, P<0>>, 3>>
//            >::Type ResultType;

//    EXPECT_TRUE((std::is_same<
//                    ResultType,
//                    Adaptive<
//                        JO<MultipleOf<O<0>, 3>>,
//                        JointProcessModel<MultipleOf<P<0>, 3>>
//                    >
//                 >::value));
//}

//TEST(MetaJoinTests, Join_Ox_ExpandAdaptive)
//{
//    typedef CollapseJoin<
//                JO<>,
//                TypeSequence<>,
//                Join<O<0>, MultipleOf<Adaptive<O<1>, P<1>>, 3>>
//            >::Type ResultType;

//    EXPECT_TRUE((std::is_same<
//                    ResultType,
//                    Adaptive<
//                        JO<NotAdaptive<O<0>>, JO<MultipleOf<O<1>, 3>>>,
//                        JointProcessModel<
//                            NAProcessModel,
//                            JointProcessModel<MultipleOf<P<1>, 3>>>
//                    >
//                 >::value));


//}

//TEST(MetaJoinTests, Join_AdaptiveOx_ExpandAdaptive)
//{
//    typedef CollapseJoin<
//                JO<>,
//                TypeSequence<>,
//                Join<Adaptive<O<0>, P<0>>, MultipleOf<Adaptive<O<1>, P<1>>, 3>>
//            >::Type ResultType;

//    EXPECT_TRUE((std::is_same<
//                    ResultType,
//                    Adaptive<
//                        JO<O<0>, JO<MultipleOf<O<1>, 3>>>,
//                        JointProcessModel<
//                            P<0>,
//                            JointProcessModel<MultipleOf<P<1>, 3>>
//                        >
//                    >
//                 >::value));
//}

//TEST(MetaJoinTests, Join_total_mix)
//{
//    typedef CollapseJoin<
//                JO<>,
//                TypeSequence<>,
//                Join<
//                    Adaptive<O<0>, P<0>>,
//                    MultipleOf<Adaptive<O<1>, P<1>>, 3>,
//                    O<2>,
//                    Join<
//                        MultipleOf<O<3>, 3>,
//                        Join<O<4>, O<5>, O<6>>,
//                        Join<
//                            Adaptive<O<7>, P<7>>,
//                            Adaptive<O<8>, P<8>>,
//                            Adaptive<O<9>, P<9>>
//                        >,
//                        Join<
//                            MultipleOf<Adaptive<O<10>, P<10>>, 1>,
//                            MultipleOf<Adaptive<O<11>, P<11>>, 2>,
//                            MultipleOf<Adaptive<O<12>, P<12>>, 3>
//                        >
//                    >
//                >
//            >::Type ResultType;

//    EXPECT_TRUE((std::is_same<
//                    ResultType,
//                    Adaptive<
//                        JO<
//                            O<0>,
//                            JO<MultipleOf<O<1>, 3>>,
//                            NotAdaptive<O<2>>,
//                            NotAdaptive<JO<MultipleOf<O<3>, 3>>>,
//                            NotAdaptive<O<4>>,
//                            NotAdaptive<O<5>>,
//                            NotAdaptive<O<6>>,
//                            O<7>, O<8>, O<9>,
//                            JO<MultipleOf<O<10>, 1>>,
//                            JO<MultipleOf<O<11>, 2>>,
//                            JO<MultipleOf<O<12>, 3>>
//                        >,
//                        JointProcessModel<
//                            P<0>,
//                            JointProcessModel<MultipleOf<P<1>, 3>>,
//                            NAProcessModel,
//                            NAProcessModel,
//                            NAProcessModel,
//                            NAProcessModel,NAProcessModel,
//                            P<7>, P<8>, P<9>,
//                            JointProcessModel<MultipleOf<P<10>, 1>>,
//                            JointProcessModel<MultipleOf<P<11>, 2>>,
//                            JointProcessModel<MultipleOf<P<12>, 3>>
//                        >
//                    >
//                 >::value));
//}


TEST(MetaJoinTests, JoinAdaptive_Ox)
{
    typedef CollapseJoin<
                JO<>,
                TypeSequence<>,
                Join<Adaptive<O<0>, P<0>>, O<1>>
            >::Type ResultType;

    EXPECT_TRUE((std::is_same<
                    ResultType,
                    Adaptive<
                        JO<O<0>, NotAdaptive<O<1>>>,
                    JointProcessModel<P<0>>>
                 >::value));
}

TEST(MetaJoinTests, Join_Ox_Adaptive)
{
    typedef CollapseJoin<
                JO<>,
                TypeSequence<>,
                Join<O<0>, Adaptive<O<1>, P<1>>>
            >::Type ResultType;

    EXPECT_TRUE((std::is_same<
                    ResultType,
                    Adaptive<
                        JO<NotAdaptive<O<0>>, O<1>>,
                        JointProcessModel<P<1>>
                    >
                 >::value));
}

TEST(MetaJoinTests, Join_Ox_Adaptive_Ox)
{
    typedef CollapseJoin<
                JO<>,
                TypeSequence<>,
                Join<O<0>, Adaptive<O<1>, P<1>>, O<2>>
            >::Type ResultType;

    EXPECT_TRUE((std::is_same<
                    ResultType,
                    Adaptive<
                        JO<NotAdaptive<O<0>>, O<1>, NotAdaptive<O<2>>>,
                        JointProcessModel<P<1>>
                        >
                 >::value));
}


TEST(MetaJoinTests, Join_Ox_Ox_Adaptive)
{
    typedef CollapseJoin<
                JO<>,
                TypeSequence<>,
                Join<O<0>, O<1>, Adaptive<O<2>, P<2>>>
            >::Type ResultType;

    EXPECT_TRUE((std::is_same<
                    ResultType,
                    Adaptive<
                        JO<NotAdaptive<O<0>>, NotAdaptive<O<1>>, O<2>>,
                        JointProcessModel<P<2>>
                        >
                 >::value));
}

TEST(MetaJoinTests, Join_Adaptive_Ox_Ox)
{
    typedef CollapseJoin<
                JO<>,
                TypeSequence<>,
                Join<Adaptive<O<0>, P<0>>, O<1>, O<2>>
            >::Type ResultType;

    EXPECT_TRUE((std::is_same<
                    ResultType,
                    Adaptive<
                        JO<O<0>, NotAdaptive<O<1>>, NotAdaptive<O<2>>>,
                        JointProcessModel<P<0>>>
                 >::value));
}

TEST(MetaJoinTests, Join_AllAdaptive)
{
    typedef CollapseJoin<
                JO<>,
                TypeSequence<>,
                Join<
                    Adaptive<O<0>, P<0>>,
                    Adaptive<O<1>, P<1>>,
                    Adaptive<O<2>, P<2>>
                >
            >::Type ResultType;

    EXPECT_TRUE((std::is_same<
                    ResultType,
                    Adaptive<
                        JO<O<0>, O<1>, O<2>>,
                        JointProcessModel<P<0>, P<1>, P<2>>
                    >
                 >::value));
}

TEST(MetaJoinTests, JoinExpandAdaptive)
{
    typedef CollapseJoin<
                JO<>,
                TypeSequence<>,
                Join<MultipleOf<Adaptive<O<0>, P<0>>, 3>>
            >::Type ResultType;

    EXPECT_TRUE((std::is_same<
                    ResultType,
                    Adaptive<
                        JO<MultipleOf<O<0>, 3>>,
                        JointProcessModel<MultipleOf<P<0>, 3>>
                    >
                 >::value));
}

TEST(MetaJoinTests, Join_Ox_ExpandAdaptive)
{
    typedef CollapseJoin<
                JO<>,
                TypeSequence<>,
                Join<O<0>, MultipleOf<Adaptive<O<1>, P<1>>, 3>>
            >::Type ResultType;

    EXPECT_TRUE((std::is_same<
                    ResultType,
                    Adaptive<
                        JO<NotAdaptive<O<0>>, JO<MultipleOf<O<1>, 3>>>,
                        JointProcessModel<MultipleOf<P<1>, 3>>
                    >
                 >::value));


}

TEST(MetaJoinTests, Join_AdaptiveOx_ExpandAdaptive)
{
    typedef CollapseJoin<
                JO<>,
                TypeSequence<>,
                Join<Adaptive<O<0>, P<0>>, MultipleOf<Adaptive<O<1>, P<1>>, 3>>
            >::Type ResultType;

    EXPECT_TRUE((std::is_same<
                    ResultType,
                    Adaptive<
                        JO<O<0>, JO<MultipleOf<O<1>, 3>>>,
                        JointProcessModel<
                            P<0>,
                            JointProcessModel<MultipleOf<P<1>, 3>>
                        >
                    >
                 >::value));
}

TEST(MetaJoinTests, Join_total_mix)
{
    typedef CollapseJoin<
                JO<>,
                TypeSequence<>,
                Join<
                    Adaptive<O<0>, P<0>>,
                    MultipleOf<Adaptive<O<1>, P<1>>, 3>,
                    O<2>,
                    Join<
                        MultipleOf<O<3>, 3>,
                        Join<O<4>, O<5>, O<6>>,
                        Join<
                            Adaptive<O<7>, P<7>>,
                            Adaptive<O<8>, P<8>>,
                            Adaptive<O<9>, P<9>>
                        >,
                        Join<
                            MultipleOf<Adaptive<O<10>, P<10>>, 1>,
                            MultipleOf<Adaptive<O<11>, P<11>>, 2>,
                            MultipleOf<Adaptive<O<12>, P<12>>, 3>
                        >
                    >
                >
            >::Type ResultType;

    EXPECT_TRUE((std::is_same<
                    ResultType,
                    Adaptive<
                        JO<
                            O<0>,
                            JO<MultipleOf<O<1>, 3>>,
                            NotAdaptive<O<2>>,
                            NotAdaptive<JO<MultipleOf<O<3>, 3>>>,
                            NotAdaptive<O<4>>,
                            NotAdaptive<O<5>>,
                            NotAdaptive<O<6>>,
                            O<7>, O<8>, O<9>,
                            JO<MultipleOf<O<10>, 1>>,
                            JO<MultipleOf<O<11>, 2>>,
                            JO<MultipleOf<O<12>, 3>>
                        >,
                        JointProcessModel<
                            P<0>,
                            JointProcessModel<MultipleOf<P<1>, 3>>,
                            P<7>, P<8>, P<9>,
                            JointProcessModel<MultipleOf<P<10>, 1>>,
                            JointProcessModel<MultipleOf<P<11>, 2>>,
                            JointProcessModel<MultipleOf<P<12>, 3>>
                        >
                    >
                 >::value));
}
