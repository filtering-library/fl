cmake_minimum_required(VERSION 2.8.3)
project(fl)

############################
# Options                  #
############################
option(fl_USE_CATKIN "Use catkin build system" ON)
option(fl_USE_RANDOM_SEED "Use random seeds for number generators" ON)
set(fl_FLOATING_POINT_TYPE "double" CACHE STRING "fl::Real floating point type")

############################
# Flags                    #
############################
# Enable c++11 GCC 4.6 or greater required
add_definitions(-DEIGEN_STACK_ALLOCATION_LIMIT=1638400)
add_definitions(-DEIGEN_MPL2_ONLY=1)
add_definitions(-std=c++0x)
add_definitions(-DPROFILING_ON=1) #print profiling output

add_definitions(-Wall)
add_definitions(-Wno-unused-local-typedefs)
add_definitions(-Wno-deprecated-declarations)
add_definitions(-Wno-comment)

# for eigen-3.1.2
add_definitions(-Wno-deprecated-register)

############################
# Setup                    #
############################
#set(fl_INCLUDE_DIRS include)
set(fl_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_MODULE_PATH ${fl_MODULE_PATH})
include(${fl_MODULE_PATH}/definitions.cmake)

############################
# Exports                 ##
############################
set(fl_FOUND TRUE PARENT_SCOPE)
set(fl_INCLUDE_DIRS ../fl/include ${Eigen_INCLUDE_DIRS} PARENT_SCOPE)
set(fl_DEFINITIONS ${Eigen_DEFINITIONS} PARENT_SCOPE)
set(fl_DEPENDS Eigen PARENT_SCOPE)

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

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

#find_package(OpenMP)
#if(OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    message (WARNING "${OpenMP_C_FLAGS} ${OpenMP_CXX_FLAGS}")
#endif()

############################
## catkin                  #
############################
if(fl_USE_CATKIN)
    find_package(catkin REQUIRED)
endif(fl_USE_CATKIN)

if(catkin_FOUND AND fl_USE_CATKIN)
    set(fl_USING_CATKIN YES)
    catkin_package(INCLUDE_DIRS include DEPENDS Eigen Boost)
    list(APPEND fl_INCLUDE_DIRS ${catkin_INCLUDE_DIRS})
endif(catkin_FOUND AND fl_USE_CATKIN)

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

info_begin()
  info_project("::fl:: Filtering Library" ${PROJECT_VERSION})
  info_header("Setup:")
  info_item("Using Catkin" "${fl_USING_CATKIN}")
  info_item("Using random seed" "${fl_USE_RANDOM_SEED}")
  info_item("Using fl::Real floating point type" ${fl_FLOATING_POINT_TYPE})
info_end()

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
