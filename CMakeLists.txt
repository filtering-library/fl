cmake_minimum_required(VERSION 2.8.3)
project(fl)

############################
# Flags                    #
############################
# Enable c++11 GCC 4.6 or greater required
add_definitions(-DEIGEN_MPL2_ONLY=1)
add_definitions(-std=c++0x -Wall -O3 -g3)
add_definitions(-DPROFILING_ON=1) #print profiling output
add_definitions(-DUSE_RANDOM_SEED=0)

############################
# Setup                    #
############################
set(fl_INCLUDE_DIRS include)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

############################
# Library Version          #
############################
include(${CMAKE_MODULE_PATH}/version.cmake)
message("##############################")
message("# Filtering Library ")
message("# Version ${PROJECT_VERSION}")
message("##############################")

############################
# Dependencies            ##
############################
find_package(Eigen 3.1 REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS})
add_definitions(${Eigen_DEFINITIONS})

############################
## catkin                 ##
############################
find_package(catkin QUIET)
if (catkin_FOUND)
    message("Using catkin ... ")
    catkin_package(INCLUDE_DIRS include DEPENDS Eigen)
    list(APPEND fl_INCLUDE_DIRS ${catkin_INCLUDE_DIRS})
endif(catkin_FOUND)

############################
# Documentation Generation #
############################
#
# How to generate the documentation:
#
#  $ cd /path/to/fl
#  $ mkdir build
#  $ cd build
#  $ cmake ..
#  $ make doc_fl
#
# The documentation will be generated within /path/to/fl/build/doc
#
set(MIN_DOXYGEN_VERSION 1.8.4)
find_package(Doxygen ${MIN_DOXYGEN_VERSION})
include(${CMAKE_MODULE_PATH}/doxygen.cmake)

############################
# Build                    #
############################
include_directories(${fl_INCLUDE_DIRS})
file(GLOB_RECURSE header_files include/${PROJECT_NAME}/*.hpp
                               include/${PROJECT_NAME}/*.h
                               include/ff/*.hpp
                               include/ff/*.h)

add_library(${PROJECT_NAME} ${header_files})
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

############################
# Tests                    #
############################
enable_testing()
include(${CMAKE_MODULE_PATH}/gtest.cmake)
add_subdirectory(test)
