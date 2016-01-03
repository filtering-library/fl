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
 * \file profiling.hpp
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#pragma once

#include <errno.h>
#include <signal.h>
#include <assert.h>
#include <sys/time.h>
#include <iostream>

// profiling macros
#define GET_TIME(time) {struct timeval profiling_time; gettimeofday(&profiling_time, NULL);\
    time = (profiling_time.tv_sec * 1000000u + profiling_time.tv_usec) /1000000.;}
#ifdef PROFILING_ON
    #define PRINT(object) std::cout << object;

    #define INIT_PROFILING struct timeval profiling_start_time, profiling_end_time;\
                                    gettimeofday(&profiling_start_time, NULL);
    #define RESET gettimeofday(&profiling_start_time, NULL);
    #define MEASURE(text)\
            gettimeofday(&profiling_end_time, NULL);\
            std::cout << "time for " << text << " " \
              << std::setprecision(9) << std::fixed\
              << ((profiling_end_time.tv_sec - profiling_start_time.tv_sec) * 1000000u\
                 + profiling_end_time.tv_usec - profiling_start_time.tv_usec) /1000000. \
              << " s" << std::endl; gettimeofday(&profiling_start_time, NULL);
    #define MEASURE_FLUSH(text)\
            gettimeofday(&profiling_end_time, NULL);\
            std::cout << "\r";\
            std::cout.flush();\
            std::cout << "time for " << text << " " \
              << std::setprecision(9) << std::fixed\
              << ((profiling_end_time.tv_sec - profiling_start_time.tv_sec) * 1000000u\
                 + profiling_end_time.tv_usec - profiling_start_time.tv_usec) /1000000. \
              << " s";  gettimeofday(&profiling_start_time, NULL);
#else
    #define PRINT(object)
    #define INIT_PROFILING
    #define RESET
    #define MEASURE(text)
#endif

#define PShape(mat) std::cout << #mat << " (" << mat.rows() << ", " << mat.cols() << ")" << "\n\n";
#define PV(mat) std::cout << #mat << "\n" << mat << "\n\n";
#define PVT(mat) std::cout << #mat << "\n" << mat.transpose() << "\n";
#define PF(flag) std::cout << #flag << ":=" << flag << "\n";
#define PInfo(text) std::cout << "Info: " << text << "\n";

#ifdef NDEBUG
#define break_on_fail(expr) (static_cast<void> (0))
#else

namespace fl
{
namespace internal
{
__attribute__ ((always_inline)) inline void __break_on_fail(
    __const char *__assertion,
    __const char *__file,
    unsigned int __line,
    __const char *__function
) __THROW
{
    std::cout << "fl::breakpoint: " << __assertion << " failed at"
              << __file << ":"
              << __line << std::endl
              << __function << " "
              << std::endl;

    raise(SIGTRAP);
}
}
}
#define break_on_fail(expr)             \
  ((expr)                               \
   ? (static_cast<void> (0))            \
   : fl::internal::__break_on_fail (    \
        __STRING(expr),                 \
        __FILE__,                       \
        __LINE__,                       \
        __PRETTY_FUNCTION__))
#endif

#define pass_or_die(expr) if (!(expr)) { std::cout << #expr << std::endl; exit(-1); }
