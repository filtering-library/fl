
# == Util tests ============================================================== #

# = trait tests = #
catkin_add_gtest(traits_test utils/traits_test.cpp gtest_main.cpp)
target_link_libraries(traits_test ${catkin_LIBRARIES})

# = meta tests = #
catkin_add_gtest(meta_test utils/meta_test.cpp gtest_main.cpp)
target_link_libraries(meta_test ${catkin_LIBRARIES})

# = options test = #
catkin_add_gtest(options_test utils/options_test.cpp gtest_main.cpp)
target_link_libraries(options_test ${catkin_LIBRARIES})

# = join tests = #
catkin_add_gtest(join_test utils/join_test.cpp gtest_main.cpp)
target_link_libraries(join_test ${catkin_LIBRARIES})

# = euler_vector test = #
catkin_add_gtest(euler_vector_test utils/euler_vector_test.cpp gtest_main.cpp)
target_link_libraries(euler_vector_test ${catkin_LIBRARIES})


# == observation model tests ================================================= #
catkin_add_gtest(linear_observation_models_test
                 model/observation/linear_observation_model_test.cpp
                 gtest_main.cpp)
target_link_libraries(linear_observation_models_test ${catkin_LIBRARIES})

# == state transition model tests ============================================ #
catkin_add_gtest(linear_state_transition_model_test
                 model/state_transition/linear_state_transition_model_test.cpp
                 gtest_main.cpp)
target_link_libraries(linear_state_transition_model_test ${catkin_LIBRARIES})

catkin_add_gtest(orientation_state_transition_function_test
                 model/state_transition/orientation_state_transition_function_test.cpp
                 gtest_main.cpp)
target_link_libraries(orientation_state_transition_function_test ${catkin_LIBRARIES})


catkin_add_gtest(binary_transition_density_test
                 model/binary_transition_density_test.cpp
                 gtest_main.cpp)
target_link_libraries(binary_transition_density_test
                      ${catkin_LIBRARIES})

# ==filter interface test ==================================================== #
catkin_add_gtest(filter_interface_test
                 filter_interface/filter_interface_test.cpp
                 filter_interface/filter_interface_stubs.hpp
                 gtest_main.cpp)
target_link_libraries(filter_interface_test
                      ${catkin_LIBRARIES})

# == distribution tests ====================================================== #
catkin_add_gtest(gaussian_distribution_test
                 distribution/gaussian_test.cpp
                 gtest_main.cpp)
target_link_libraries(gaussian_distribution_test
                      ${catkin_LIBRARIES})

catkin_add_gtest(discrete_distribution_test
              distribution/discrete_distribution_test.cpp
              gtest_main.cpp)
target_link_libraries(discrete_distribution_test
                   ${catkin_LIBRARIES})

# == exceptions tests ======================================================== #
catkin_add_gtest(exception_test
                 exception/exception_test.cpp
                 gtest_main.cpp)
target_link_libraries(exception_test ${catkin_LIBRARIES})

# == Gaussian filters tests ================================================== #

# = Kalman filter = #
catkin_add_gtest(gaussian_filter_kalman_filter_test
                 typecast.hpp
                 gaussian_filter/gaussian_filter_test_suite.hpp
                 gaussian_filter/kalman_filter_test.cpp
                 gtest_main.cpp)
target_link_libraries(gaussian_filter_kalman_filter_test ${catkin_LIBRARIES})

# = Unscented Kalman filter = #
catkin_add_gtest(gaussian_filter_unscented_kalman_filter_test
                 typecast.hpp
                 gaussian_filter/gaussian_filter_test_suite.hpp
                 gaussian_filter/unscented_kalman_filter_test.cpp
                 gtest_main.cpp)
target_link_libraries(gaussian_filter_unscented_kalman_filter_test
                      ${catkin_LIBRARIES})

# == Particle filters tests ================================================== #
#catkin_add_gtest(particle_filter_test
#                 particle_filter/particle_filter_test.cpp
#                 gtest_main.cpp)
#target_link_libraries(particle_filter_test
#                      ${catkin_LIBRARIES})

catkin_add_gtest(particle_filter_fixture_test
                 particle_filter/particle_filter_fixture_test.cpp
                 gtest_main.cpp)
target_link_libraries(particle_filter_fixture_test
                      ${catkin_LIBRARIES})

#catkin_add_gtest(frb_particle_filter_test
#                 particle_filter/frb_particle_filter_test.cpp
#                 gtest_main.cpp)
#target_link_libraries(frb_particle_filter_test
#                      ${catkin_LIBRARIES})


# == require refactoring ===================================================== #


### some quick tests ##
##catkin_add_gtest(some_test
##                 utils/some_test.cpp
##                 gtest_main.cpp)
##target_link_libraries(some_test
##                      ${catkin_LIBRARIES})






#catkin_add_gtest(not_adaptive_test utils/not_adaptive_test.cpp gtest_main.cpp)
#target_link_libraries(not_adaptive_test ${catkin_LIBRARIES})




# ## point_set_test tests ##
# catkin_add_gtest(point_set_test
#                  gaussian_filter/point_set_test.cpp
#                  gtest_main.cpp)
# target_link_libraries(point_set_test
#                       ${catkin_LIBRARIES})

# ## unscented_transform tests ##
# catkin_add_gtest(unscented_transform_test
#                  gaussian_filter/unscented_transform_test.cpp
#                  gtest_main.cpp)
# target_link_libraries(unscented_transform_test
#                       ${catkin_LIBRARIES})

# ## monte_carlo trandform tests ##
## catkin_add_gtest(monte_carlo_transform_test
##                  gaussian_filter/monte_carlo_transform_test.cpp
##                  gtest_main.cpp)
## target_link_libraries(monte_carlo_transform_test
##                       ${catkin_LIBRARIES})



# catkin_add_gtest(joint_distribution_id_test
#                  distribution/joint_distribution_id_test.cpp
#                  gtest_main.cpp)
# target_link_libraries(joint_distribution_id_test ${catkin_LIBRARIES})


# catkin_add_gtest(joint_distribution_iid_test
#                  distribution/joint_distribution_iid_test.cpp
#                  gtest_main.cpp)
# target_link_libraries(joint_distribution_iid_test ${catkin_LIBRARIES})





# catkin_add_gtest(joint_observation_model_iid_test
#                  model/joint_observation_model_iid_test.cpp
#                  gtest_main.cpp)
# target_link_libraries(joint_observation_model_iid_test ${catkin_LIBRARIES})

# catkin_add_gtest(joint_process_model_id_test
#                  model/joint_process_model_id_test.cpp
#                  gtest_main.cpp)
# target_link_libraries(joint_process_model_id_test ${catkin_LIBRARIES})


# catkin_add_gtest(joint_process_model_iid_test
#                  model/joint_process_model_iid_test.cpp
#                  gtest_main.cpp)
# target_link_libraries(joint_process_model_iid_test ${catkin_LIBRARIES})


# Factorized SPKF #
#catkin_add_gtest(gaussian_filter_factorized_test
#                 gaussian_filter/gaussian_filter_factorized_test.cpp
#                 gtest_main.cpp)
#target_link_libraries(gaussian_filter_factorized_test ${catkin_LIBRARIES})







#catkin_add_gtest(adaptive_test
#                 model/adaptive_test.cpp
#                 gtest_main.cpp)
#target_link_libraries(adaptive_test ${catkin_LIBRARIES})


# ## Gaussian filter tests ##
# catkin_add_gtest(gaussian_filter_test
#                  gaussian_filter/gaussian_filter_test.cpp
#                  gaussian_filter/gaussian_filter_stubs.hpp
#                  gtest_main.cpp)
# target_link_libraries(gaussian_filter_test
#                       ${catkin_LIBRARIES})


## Factorized UKF vs. Kalman filter tests ##
# catkin_add_gtest(factorized_ukf_kf_test
#                  factorized_ukf/factorized_ukf_kf_test.cpp
#                  gtest_main.cpp)
# target_link_libraries(factorized_ukf_kf_test
#                       ${catkin_LIBRARIES})
##
## ## Factorized UKF tests ##
# catkin_add_gtest(factorized_ukf_test
##                  factorized_ukf/linear_models_test.cpp
#                  factorized_ukf/partitioned_ut_test.cpp
#                  factorized_ukf/smw_inversion_test.cpp
#                  factorized_ukf/square_root_test.cpp
##                  factorized_ukf/llt_diagonal_test.cpp
#                  factorized_ukf/factorized_ukf_unscented_transform_test.cpp
##                  factorized_ukf/predict_test.cpp
#                  factorized_ukf/factorized_ukf_dummies.hpp
#                  factorized_ukf/factorized_ukf_stubs.hpp
#                  gtest_main.cpp)
# target_link_libraries(factorized_ukf_test
#                       ${PROJECT_NAME_TRACKING_INTERFACE}
#                       ${PROJECT_NAME_TRACKING}
#                       ${catkin_LIBRARIES})
