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
 * \file profiling.hpp
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#ifndef FL__UTIL__PROFILING_HPP
#define FL__UTIL__PROFILING_HPP

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
#else
    #define PRINT(object)
    #define INIT_PROFILING
    #define RESET
    #define MEASURE(text)
#endif

#define PV(mat) std::cout << #mat << "\n" << mat << "\n\n";
//#define PV(mat)

#endif
