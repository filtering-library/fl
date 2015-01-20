cmake_minimum_required(VERSION 2.8.3)
project(fl)

############################
# Flags                    #
############################
# Enable c++11 GCC 4.6 or greater required
add_definitions(-std=c++0x)

add_definitions(-DPROFILING_ON=1) #print profiling output
add_definitions(-DUSE_RANDOM_SEED=1)

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

############################
# Library Version          #
############################
set(FL_MAJOR_VERSION 0)
set(FL_MINOR_VERSION 1)
execute_process(COMMAND git rev-list --count HEAD
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                OUTPUT_VARIABLE FL_BUILD_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND git rev-parse --short HEAD
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                OUTPUT_VARIABLE FL_REV_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE)

message("##############################")
message("# Filtering Library ")
message("# Version ${FL_MAJOR_VERSION}.${FL_MINOR_VERSION}.${FL_BUILD_VERSION}")
message("##############################")

############################
# Dependecies              #
############################
find_package(catkin REQUIRED)
find_package(Eigen REQUIRED)

include_directories(${Eigen_INCLUDE_DIRS})

############################
## catkin                 ##
############################
catkin_package(
    INCLUDE_DIRS include
    DEPENDS Eigen
)

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
#  $ make doc
#
# The documentation will be generated within /path/to/fl/build/doc
#
find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(doc_fl
        ${CMAKE_COMMAND} ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif(DOXYGEN_FOUND)

set(PROJECT_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_MODULE_PATH ${PROJECT_MODULE_PATH})

############################
# Build                    #
############################
include_directories(include ${catkin_INCLUDE_DIRS})
file(GLOB_RECURSE header_files include/${PROJECT_NAME}/*.hpp
                               include/${PROJECT_NAME}/*.h
                               include/ff/*.hpp
                               include/ff/*.h)

add_library(${PROJECT_NAME} ${header_files})
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

############################
# Tests                    #
############################
add_subdirectory(test)
