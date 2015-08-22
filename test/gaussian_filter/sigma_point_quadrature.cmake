
# add_gaussian_filter_vs_x tests #
function(add_sigma_point_quadrature_test
    Name
    Type
    DimA
    DimB
    Transform)

    set(TEST_FILE ${Name})

    set(TEST_FILE_SUFFIX "${Type}_${DimA}_${DimB}_${Transform}")

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_FILE}.cpp.in
        ${CMAKE_CURRENT_BINARY_DIR}/${TEST_FILE}_${TEST_FILE_SUFFIX}.cpp @ONLY)

    fl_add_test(
        NAME    ${TEST_FILE}_${TEST_FILE_SUFFIX}
        SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_FILE}.hpp
                ${CMAKE_CURRENT_BINARY_DIR}/${TEST_FILE}_${TEST_FILE_SUFFIX}.cpp)
endfunction()

set(CurrentTest "sigma_point_quadrature_test")

add_sigma_point_quadrature_test(${CurrentTest} StaticTest 1 1 Unscented)
add_sigma_point_quadrature_test(${CurrentTest} StaticTest 2 2 Unscented)
add_sigma_point_quadrature_test(${CurrentTest} StaticTest 3 3 Unscented)
add_sigma_point_quadrature_test(${CurrentTest} StaticTest 6 3 Unscented)
add_sigma_point_quadrature_test(${CurrentTest} StaticTest 3 6 Unscented)
add_sigma_point_quadrature_test(${CurrentTest} StaticTest 24 3 Unscented)
add_sigma_point_quadrature_test(${CurrentTest} StaticTest 50 50 Unscented)

add_sigma_point_quadrature_test(${CurrentTest} DynamicTest 1 1 Unscented)
add_sigma_point_quadrature_test(${CurrentTest} DynamicTest 2 2 Unscented)
add_sigma_point_quadrature_test(${CurrentTest} DynamicTest 3 3 Unscented)
add_sigma_point_quadrature_test(${CurrentTest} DynamicTest 6 3 Unscented)
add_sigma_point_quadrature_test(${CurrentTest} DynamicTest 3 6 Unscented)
add_sigma_point_quadrature_test(${CurrentTest} DynamicTest 24 3 Unscented)
add_sigma_point_quadrature_test(${CurrentTest} DynamicTest 50 50 Unscented)

add_sigma_point_quadrature_test(${CurrentTest} StaticTest 1 1 MonteCarlo)
add_sigma_point_quadrature_test(${CurrentTest} StaticTest 2 2 MonteCarlo)
add_sigma_point_quadrature_test(${CurrentTest} StaticTest 3 3 MonteCarlo)
add_sigma_point_quadrature_test(${CurrentTest} StaticTest 6 3 MonteCarlo)
add_sigma_point_quadrature_test(${CurrentTest} StaticTest 3 6 MonteCarlo)
add_sigma_point_quadrature_test(${CurrentTest} StaticTest 24 3 MonteCarlo)
add_sigma_point_quadrature_test(${CurrentTest} StaticTest 50 50 MonteCarlo)

add_sigma_point_quadrature_test(${CurrentTest} DynamicTest 1 1 MonteCarlo)
add_sigma_point_quadrature_test(${CurrentTest} DynamicTest 2 2 MonteCarlo)
add_sigma_point_quadrature_test(${CurrentTest} DynamicTest 3 3 MonteCarlo)
add_sigma_point_quadrature_test(${CurrentTest} DynamicTest 6 3 MonteCarlo)
add_sigma_point_quadrature_test(${CurrentTest} DynamicTest 3 6 MonteCarlo)
add_sigma_point_quadrature_test(${CurrentTest} DynamicTest 24 3 MonteCarlo)
add_sigma_point_quadrature_test(${CurrentTest} DynamicTest 50 50 MonteCarlo)
