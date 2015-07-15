cmake_minimum_required(VERSION 2.8.3)
project(fl)

############################
# Flags                    #
############################
# Enable c++11 GCC 4.6 or greater required
add_definitions(-std=c++0x -Wall -O3 -g3)

add_definitions(-DPROFILING_ON=1) #print profiling output
add_definitions(-DUSE_RANDOM_SEED=0)

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
# Setup                    #
############################
set(PROJECT_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_MODULE_PATH ${PROJECT_MODULE_PATH})

############################
# Dependecies
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
set(MIN_DOXYGEN_VERSION 1.8.4)
find_package(Doxygen ${MIN_DOXYGEN_VERSION})

set(TARGET_FAILED_SCRIPT_TEMPLATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/target_failed.cmake.in)

set(TARGET_FAILED_SCRIPT
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/target_failed.cmake.in)

if(DOXYGEN_FOUND)
    execute_process(COMMAND "${DOXYGEN_EXECUTABLE}" "--version"
                    OUTPUT_VARIABLE DOXYGEN_VERSION
                    OUTPUT_STRIP_TRAILING_WHITESPACE)

    if( "${DOXYGEN_VERSION}" VERSION_LESS "${MIN_DOXYGEN_VERSION}" )
        set(DOXYGEN_WARN_MSG_OLD   "Doxygen version is too old!")
        set(DOXYGEN_WARN_MSG_FOUND "Found Doxygen ${DOXYGEN_VERSION}.")
        set(DOXYGEN_WARN_MSG_REQ   "Required is at least ${MIN_DOXYGEN_VERSION}")

        set(DOXYGEN_WARN_MSG "\n${DOXYGEN_WARN_MSG_OLD}\n")
        set(DOXYGEN_WARN_MSG "${DOXYGEN_WARN_MSG} (${DOXYGEN_WARN_MSG_FOUND}")
        set(DOXYGEN_WARN_MSG "${DOXYGEN_WARN_MSG} ${DOXYGEN_WARN_MSG_REQ})")
        message(WARNING ${DOXYGEN_WARN_MSG})

        set(FATAL_ERROR_MESSAGE ${DOXYGEN_WARN_MSG})
        configure_file(
            ${TARGET_FAILED_SCRIPT_TEMPLATE}
            ${TARGET_FAILED_SCRIPT} @ONLY)

        add_custom_target(doc_fl
            COMMAND ${CMAKE_COMMAND} -P ${TARGET_FAILED_SCRIPT})
    else()
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in
            ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
        add_custom_target(doc_fl
            ${CMAKE_COMMAND} ${CMAKE_CURRENT_SOURCE_DIR}
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen" VERBATIM)
    endif()
else(DOXYGEN_FOUND)
    set(DOXYGEN_WARN_MSG "Doxygen not found.")
    message(WARNING ${DOXYGEN_WARN_MSG})

    set(FATAL_ERROR_MESSAGE ${DOXYGEN_WARN_MSG})
    configure_file(
        ${TARGET_FAILED_SCRIPT_TEMPLATE}
        ${TARGET_FAILED_SCRIPT} @ONLY)

    add_custom_target(doc_fl
        COMMAND ${CMAKE_COMMAND} -P ${TARGET_FAILED_SCRIPT})
endif(DOXYGEN_FOUND)


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
