
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include(gaussian_filter.cmake)
include(sigma_point_quadrature.cmake)

add_executable(
    robust_gaussian_filter_plot_data
    robust_gaussian_filter_plot_data.cpp)
target_link_libraries(robust_gaussian_filter_plot_data ${${PROJECT_NAME}_LIBS})
