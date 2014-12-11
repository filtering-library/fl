cmake_minimum_required(VERSION 2.8.3)
project(fl)

#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)
add_definitions(-DPROFILING_ON=1) #print profiling output
add_definitions(-std=c++0x)

find_package(catkin REQUIRED)

# library version
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

############################
# documentation generation #
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
    add_custom_target(doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif(DOXYGEN_FOUND)

set(PROJECT_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_MODULE_PATH ${PROJECT_MODULE_PATH})

find_package(Eigen REQUIRED)
find_package(Boost REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${Eigen_INCLUDE_DIRS})

###################################
## catkin specific configuration ##
###################################
catkin_package(
    INCLUDE_DIRS include
    # LIBRARIES 
    # CATKIN_DEPENDS 
    DEPENDS Boost Eigen
)

###########
## Build ##
###########

include_directories(include ${catkin_INCLUDE_DIRS})
file(GLOB_RECURSE header_files include/${PROJECT_NAME}/*.hpp 
                               include/${PROJECT_NAME}/*.h)

add_library(${PROJECT_NAME} ${header_files})
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

###########
## Tests ##
###########

# trait tests ##
catkin_add_gtest(traits_tests
                 tests/utils/traits_test.cpp
                 tests/gtest_main.cpp)
target_link_libraries(traits_tests
                      ${catkin_LIBRARIES})
# 
# ## filter interface tests ##
# catkin_add_gtest(filter_interface_tests
#                  tests/filter_interface/filter_interface_test.cpp
#                  tests/filter_interface/filter_interface_stubs.hpp
#                  tests/gtest_main.cpp)
# target_link_libraries(filter_interface_tests
#                       ${catkin_LIBRARIES})
# 
# ## Gaussian filter tests ##
# catkin_add_gtest(gaussian_filter_tests
#                  tests/gaussian_filter/gaussian_filter_test.cpp
#                  tests/gaussian_filter/gaussian_filter_stubs.hpp
#                  tests/gtest_main.cpp)
# target_link_libraries(gaussian_filter_tests
#                       ${catkin_LIBRARIES})
# 
# ## point_set_tests tests ##
# catkin_add_gtest(point_set_tests
#                  tests/gaussian_filter/point_set_test.cpp
#                  tests/gtest_main.cpp)
# target_link_libraries(point_set_tests
#                       ${catkin_LIBRARIES})
# 
# ## unscented_transform tests ##
# catkin_add_gtest(unscented_transform_tests
#                  tests/gaussian_filter/unscented_transform_test.cpp
#                  tests/gtest_main.cpp)
# target_link_libraries(unscented_transform_tests
#                       ${catkin_LIBRARIES})
# 
# ## various filter tests ##
# catkin_add_gtest(distribution_tests
#                  tests/distribution/gaussian_test.cpp
#                  tests/gtest_main.cpp)
# target_link_libraries(distribution_tests
#                       ${catkin_LIBRARIES})
# 
# ## exceptions tests ##
# catkin_add_gtest(exception_tests
#                  tests/exception/exception_test.cpp
#                  tests/gtest_main.cpp)
# target_link_libraries(exception_tests
#                       ${catkin_LIBRARIES})
# 
# 
# ## Kalman filter tests ##
# catkin_add_gtest(kalman_filter_tests
#                  tests/kalman_filter/kalman_filter_test.cpp
#                  tests/gtest_main.cpp)
# target_link_libraries(kalman_filter_tests
#                       ${catkin_LIBRARIES})
# 
# ## linear models tests ##
# catkin_add_gtest(linear_models_tests
#                  tests/linear_model/linear_process_model_test.cpp
#                  tests/linear_model/linear_observation_model_test.cpp
#                  tests/gtest_main.cpp)
# target_link_libraries(linear_models_tests
#                       ${catkin_LIBRARIES})
# 
# ## Factorized UKF vs. Kalman filter tests ##
# catkin_add_gtest(factorized_ukf_kf_tests
#                  tests/factorized_ukf/factorized_ukf_kf_test.cpp
#                  tests/gtest_main.cpp)
# target_link_libraries(factorized_ukf_kf_tests
#                       ${catkin_LIBRARIES})
# 
# ## Factorized UKF tests ##
# catkin_add_gtest(factorized_ukf_tests
#                  tests/factorized_ukf/linear_models_test.cpp
#                  tests/factorized_ukf/partitioned_ut_test.cpp
#                  tests/factorized_ukf/smw_inversion_test.cpp
#                  tests/factorized_ukf/square_root_test.cpp
#                  tests/factorized_ukf/llt_diagonal_test.cpp
#                  tests/factorized_ukf/factorized_ukf_unscented_transform_test.cpp
#                  tests/factorized_ukf/predict_test.cpp
#                  tests/factorized_ukf/factorized_ukf_dummies.hpp
#                  tests/factorized_ukf/factorized_ukf_stubs.hpp
#                  tests/factorized_ukf/continuous_kinect_pixel_observation_model_test.cpp
#                  tests/gtest_main.cpp)
# target_link_libraries(factorized_ukf_tests
#                       ${PROJECT_NAME_TRACKING_INTERFACE}
#                       ${PROJECT_NAME_TRACKING}
#                       ${catkin_LIBRARIES})
