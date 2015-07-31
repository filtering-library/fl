
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

add_gaussian_filter_test(StaticTest  2  4  2  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  3  4  2  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  4  4  2  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  5  4  2  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  10 4  2  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  20 4  2  GaussianModel NonAdditive NonAdditive)

add_gaussian_filter_test(StaticTest  3  4  3  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  4  4  4  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  5  4  5  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  10 4  10  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  20 4  20  GaussianModel NonAdditive NonAdditive)

add_gaussian_filter_test(StaticTest  2  4  3  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  2  4  4  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  2  4  5  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  2  4  10  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  2  4  20  GaussianModel NonAdditive NonAdditive)

add_gaussian_filter_test(DynamicTest  3  4  2  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  4  4  2  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  5  4  2  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  10 4  2  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  20 4  2  GaussianModel NonAdditive NonAdditive)

add_gaussian_filter_test(DynamicTest  3  4  3  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  4  4  4  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  5  4  5  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  10 4  10  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  20 4  20  GaussianModel NonAdditive NonAdditive)

add_gaussian_filter_test(DynamicTest  2  4  3  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  2  4  4  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  2  4  5  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  2  4  10  GaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  2  4  20  GaussianModel NonAdditive NonAdditive)

add_gaussian_filter_test(StaticTest  2  4  2  GaussianModel Additive Additive)
add_gaussian_filter_test(StaticTest  3  4  2  GaussianModel Additive Additive)
add_gaussian_filter_test(StaticTest  4  4  2  GaussianModel Additive Additive)
add_gaussian_filter_test(StaticTest  5  4  2  GaussianModel Additive Additive)
add_gaussian_filter_test(StaticTest  10 4  2  GaussianModel Additive Additive)
add_gaussian_filter_test(StaticTest  20 4  2  GaussianModel Additive Additive)

add_gaussian_filter_test(StaticTest  3  4  3  GaussianModel Additive Additive)
add_gaussian_filter_test(StaticTest  4  4  4  GaussianModel Additive Additive)
add_gaussian_filter_test(StaticTest  5  4  5  GaussianModel Additive Additive)
add_gaussian_filter_test(StaticTest  10 4  10  GaussianModel Additive Additive)
add_gaussian_filter_test(StaticTest  20 4  20  GaussianModel Additive Additive)

add_gaussian_filter_test(StaticTest  2  4  3  GaussianModel Additive Additive)
add_gaussian_filter_test(StaticTest  2  4  4  GaussianModel Additive Additive)
add_gaussian_filter_test(StaticTest  2  4  5  GaussianModel Additive Additive)
add_gaussian_filter_test(StaticTest  2  4  10  GaussianModel Additive Additive)
add_gaussian_filter_test(StaticTest  2  4  20  GaussianModel Additive Additive)

add_gaussian_filter_test(DynamicTest  3  4  2  GaussianModel Additive Additive)
add_gaussian_filter_test(DynamicTest  4  4  2  GaussianModel Additive Additive)
add_gaussian_filter_test(DynamicTest  5  4  2  GaussianModel Additive Additive)
add_gaussian_filter_test(DynamicTest  10 4  2  GaussianModel Additive Additive)
add_gaussian_filter_test(DynamicTest  20 4  2  GaussianModel Additive Additive)

add_gaussian_filter_test(DynamicTest  3  4  3  GaussianModel Additive Additive)
add_gaussian_filter_test(DynamicTest  4  4  4  GaussianModel Additive Additive)
add_gaussian_filter_test(DynamicTest  5  4  5  GaussianModel Additive Additive)
add_gaussian_filter_test(DynamicTest  10 4  10  GaussianModel Additive Additive)
add_gaussian_filter_test(DynamicTest  20 4  20  GaussianModel Additive Additive)

add_gaussian_filter_test(DynamicTest  2  4  3  GaussianModel Additive Additive)
add_gaussian_filter_test(DynamicTest  2  4  4  GaussianModel Additive Additive)
add_gaussian_filter_test(DynamicTest  2  4  5  GaussianModel Additive Additive)
add_gaussian_filter_test(DynamicTest  2  4  10  GaussianModel Additive Additive)
add_gaussian_filter_test(DynamicTest  2  4  20  GaussianModel Additive Additive)

add_gaussian_filter_test(StaticTest  2  4  2  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  3  4  2  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  4  4  2  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  5  4  2  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  10 4  2  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  20 4  2  DecorrelatedGaussianModel NonAdditive NonAdditive)

add_gaussian_filter_test(StaticTest  3  4  3  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  4  4  4  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  5  4  5  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  10 4  10  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  20 4  20  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  20 4  200  DecorrelatedGaussianModel NonAdditive AdditiveUncorrelated)

add_gaussian_filter_test(StaticTest  2  4  3  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  2  4  4  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  2  4  5  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  2  4  10  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  2  4  20  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  2  4  200  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(StaticTest  2  4  500  DecorrelatedGaussianModel NonAdditive AdditiveUncorrelated)
add_gaussian_filter_test(StaticTest  2  4  1000  DecorrelatedGaussianModel NonAdditive AdditiveUncorrelated)

add_gaussian_filter_test(DynamicTest  3  4  2  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  4  4  2  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  5  4  2  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  10 4  2  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  20 4  2  DecorrelatedGaussianModel NonAdditive NonAdditive)

add_gaussian_filter_test(DynamicTest  3  4  3  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  4  4  4  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  5  4  5  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  10 4  10  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  20 4  20  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  20 4  200  DecorrelatedGaussianModel NonAdditive AdditiveUncorrelated)

add_gaussian_filter_test(DynamicTest  2  4  3  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  2  4  4  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  2  4  5  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  2  4  10  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  2  4  20  DecorrelatedGaussianModel NonAdditive NonAdditive)
add_gaussian_filter_test(DynamicTest  2  4  200  DecorrelatedGaussianModel NonAdditive AdditiveUncorrelated)
add_gaussian_filter_test(DynamicTest  2  4  500  DecorrelatedGaussianModel NonAdditive AdditiveUncorrelated)
add_gaussian_filter_test(DynamicTest  2  4  1000  DecorrelatedGaussianModel NonAdditive AdditiveUncorrelated)


add_gaussian_filter_test(StaticTest  2  4  2  DecorrelatedGaussianModel Additive Additive)
add_gaussian_filter_test(StaticTest  3  4  2  DecorrelatedGaussianModel Additive AdditiveUncorrelated)
add_gaussian_filter_test(StaticTest  4  4  2  DecorrelatedGaussianModel Additive Additive)
add_gaussian_filter_test(StaticTest  5  4  2  DecorrelatedGaussianModel Additive Additive)
add_gaussian_filter_test(StaticTest  10 4  2  DecorrelatedGaussianModel Additive AdditiveUncorrelated)
add_gaussian_filter_test(StaticTest  20 4  2  DecorrelatedGaussianModel Additive Additive)

add_gaussian_filter_test(StaticTest  3  4  3  DecorrelatedGaussianModel Additive Additive)
add_gaussian_filter_test(StaticTest  4  4  4  DecorrelatedGaussianModel Additive Additive)
add_gaussian_filter_test(StaticTest  5  4  5  DecorrelatedGaussianModel Additive AdditiveUncorrelated)
add_gaussian_filter_test(StaticTest  10 4  10  DecorrelatedGaussianModel Additive Additive)
add_gaussian_filter_test(StaticTest  20 4  20  DecorrelatedGaussianModel Additive Additive)
add_gaussian_filter_test(StaticTest  20 4  200  DecorrelatedGaussianModel Additive AdditiveUncorrelated)

add_gaussian_filter_test(StaticTest  2  4  3  DecorrelatedGaussianModel Additive Additive)
add_gaussian_filter_test(StaticTest  2  4  4  DecorrelatedGaussianModel Additive AdditiveUncorrelated)
add_gaussian_filter_test(StaticTest  2  4  5  DecorrelatedGaussianModel Additive Additive)
add_gaussian_filter_test(StaticTest  2  4  10  DecorrelatedGaussianModel Additive Additive)
add_gaussian_filter_test(StaticTest  2  4  20  DecorrelatedGaussianModel Additive Additive)
add_gaussian_filter_test(StaticTest  2  4  200  DecorrelatedGaussianModel Additive AdditiveUncorrelated)
add_gaussian_filter_test(StaticTest  2  4  500  DecorrelatedGaussianModel Additive AdditiveUncorrelated)
add_gaussian_filter_test(StaticTest  2  4  1000  DecorrelatedGaussianModel Additive AdditiveUncorrelated)

add_gaussian_filter_test(DynamicTest  3  4  2  DecorrelatedGaussianModel Additive Additive)
add_gaussian_filter_test(DynamicTest  4  4  2  DecorrelatedGaussianModel Additive AdditiveUncorrelated)
add_gaussian_filter_test(DynamicTest  5  4  2  DecorrelatedGaussianModel Additive Additive)
add_gaussian_filter_test(DynamicTest  10 4  2  DecorrelatedGaussianModel Additive Additive)
add_gaussian_filter_test(DynamicTest  20 4  2  DecorrelatedGaussianModel Additive Additive)

add_gaussian_filter_test(DynamicTest  3  4  3  DecorrelatedGaussianModel Additive Additive)
add_gaussian_filter_test(DynamicTest  4  4  4  DecorrelatedGaussianModel Additive AdditiveUncorrelated)
add_gaussian_filter_test(DynamicTest  5  4  5  DecorrelatedGaussianModel Additive Additive)
add_gaussian_filter_test(DynamicTest  10 4  10  DecorrelatedGaussianModel Additive Additive)
add_gaussian_filter_test(DynamicTest  20 4  20  DecorrelatedGaussianModel Additive Additive)
add_gaussian_filter_test(DynamicTest  20 4  200  DecorrelatedGaussianModel Additive AdditiveUncorrelated)

add_gaussian_filter_test(DynamicTest  2  4  3  DecorrelatedGaussianModel Additive Additive)
add_gaussian_filter_test(DynamicTest  2  4  4  DecorrelatedGaussianModel Additive Additive)
add_gaussian_filter_test(DynamicTest  2  4  5  DecorrelatedGaussianModel Additive AdditiveUncorrelated)
add_gaussian_filter_test(DynamicTest  2  4  10  DecorrelatedGaussianModel Additive Additive)
add_gaussian_filter_test(DynamicTest  2  4  20  DecorrelatedGaussianModel Additive Additive)
add_gaussian_filter_test(DynamicTest  2  4  200  DecorrelatedGaussianModel Additive AdditiveUncorrelated)
add_gaussian_filter_test(DynamicTest  2  4  500  DecorrelatedGaussianModel Additive AdditiveUncorrelated)
add_gaussian_filter_test(DynamicTest  2  4  1000  DecorrelatedGaussianModel Additive AdditiveUncorrelated)
