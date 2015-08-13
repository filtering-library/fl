
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

function(add_gaussian_filter_test
    Type
    StateDim
    InputDim
    ObsrvDim
    ModelType
    StateTransitionModelNoiseType
    ObsrvModelNoiseType)

    set(TEST_FILE "gaussian_filter_linear_vs_unscented_kalman_filter_test")

    set(TEST_FILE_SUFFIX "${Type}_${StateDim}_${InputDim}_${ObsrvDim}_${ModelType}_${StateTransitionModelNoiseType}_${ObsrvModelNoiseType}")

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_FILE}.cpp.in
        ${CMAKE_CURRENT_BINARY_DIR}/${TEST_FILE}_${TEST_FILE_SUFFIX}.cpp @ONLY)

    fl_add_test(
        NAME    ${TEST_FILE}_${TEST_FILE_SUFFIX}
        SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_FILE}.hpp
                ${CMAKE_CURRENT_BINARY_DIR}/${TEST_FILE}_${TEST_FILE_SUFFIX}.cpp)
endfunction()

add_gaussian_filter_test(StaticTest  2  1  2  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  20 2  2  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  20 4  20  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  2  4  20  GaussianModel NonAdditive NonAdditive)

add_gaussian_filter_test(DynamicTest  2  1  2  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  20 4  2  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  20 4  20  GaussianModel NonAdditive NonAdditive)

add_gaussian_filter_test(StaticTest  2  1  2  GaussianModel Additive Additive)
add_gaussian_filter_test(StaticTest  2  2  2  GaussianModel Additive Additive)
add_gaussian_filter_test(StaticTest  20 4  2  GaussianModel Additive Additive)

add_gaussian_filter_test(StaticTest  3  4  3  GaussianModel Additive Additive)
add_gaussian_filter_test(StaticTest  20 4  20  GaussianModel Additive Additive)
add_gaussian_filter_test(StaticTest  2  4  20  GaussianModel Additive Additive)

add_gaussian_filter_test(DynamicTest  20 4  2  GaussianModel Additive Additive)
add_gaussian_filter_test(DynamicTest  20 4  20  GaussianModel Additive Additive)
add_gaussian_filter_test(DynamicTest  2  4  20  GaussianModel Additive Additive)

add_gaussian_filter_test(StaticTest  20 4  2  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  20 4  200  DecorrelatedGaussianModel NonAdditive AdditiveUncorrelated)
add_gaussian_filter_test(StaticTest  2  4  200  DecorrelatedGaussianModel NonAdditive NonAdditive)

add_gaussian_filter_test(DynamicTest  20 4  2  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  20 4  20  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  20 4  200  DecorrelatedGaussianModel NonAdditive AdditiveUncorrelated)
add_gaussian_filter_test(DynamicTest  2  4  200  DecorrelatedGaussianModel NonAdditive AdditiveUncorrelated)

add_gaussian_filter_test(StaticTest  3  4  2  DecorrelatedGaussianModel Additive AdditiveUncorrelated)
add_gaussian_filter_test(StaticTest  10 4  2  DecorrelatedGaussianModel Additive AdditiveUncorrelated)

add_gaussian_filter_test(DynamicTest  4  4  4  DecorrelatedGaussianModel Additive AdditiveUncorrelated)
add_gaussian_filter_test(DynamicTest  20 4  200  DecorrelatedGaussianModel Additive AdditiveUncorrelated)
add_gaussian_filter_test(DynamicTest  2  4  200  DecorrelatedGaussianModel Additive AdditiveUncorrelated)
