cmake_minimum_required(VERSION 2.8.3)
project(fl)

############################
# Options                  #
############################
option(fl_USE_CATKIN "Use catkin build system" OFF)
option(fl_USE_RANDOM_SEED "Use random seeds for number generators" ON)
set(fl_FLOATING_POINT_TYPE "double" CACHE STRING "fl::Real floating point type")

############################
# Flags                    #
############################
# Enable c++11 GCC 4.6 or greater required
add_definitions(-DEIGEN_MPL2_ONLY=1)
add_definitions(-std=c++0x -O3)
add_definitions(-DPROFILING_ON=1) #print profiling output

add_definitions(-Wall)
add_definitions(-Wno-unused-local-typedefs)
add_definitions(-Wno-deprecated-declarations)
add_definitions(-Wno-comment)

############################
# Setup                    #
############################
set(fl_INCLUDE_DIRS include)
set(fl_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_MODULE_PATH ${fl_MODULE_PATH})
include(${fl_MODULE_PATH}/definitions.cmake)

############################
## catkin                  #
############################
if(fl_USE_CATKIN)
    find_package(catkin REQUIRED)
endif(fl_USE_CATKIN)

if(catkin_FOUND AND fl_USE_CATKIN)
    set(fl_USING_CATKIN YES)
    catkin_package(INCLUDE_DIRS include ${Eigen_INCLUDE_DIRS} DEPENDS Eigen)
    list(APPEND fl_INCLUDE_DIRS ${catkin_INCLUDE_DIRS})
endif(catkin_FOUND AND fl_USE_CATKIN)

############################
# Library Version          #
############################
include(${fl_MODULE_PATH}/version.cmake)

############################
# Dependencies            ##
############################
find_package(Eigen REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS})
add_definitions(${Eigen_DEFINITIONS})

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
include(${fl_MODULE_PATH}/doxygen.cmake)

############################
# Library info summary     #
############################
include(${fl_MODULE_PATH}/info.cmake)

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
include(${fl_MODULE_PATH}/gtest.cmake)
add_subdirectory(test)
