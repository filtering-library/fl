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

set(DOC_SYNC_LOCATION ""
    CACHE STRING "Sync location (URL or PATH where to sync the doc to.)")

set(TARGET_FAILED_SCRIPT_TEMPLATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/target_failed.cmake.in)

set(TARGET_FAILED_SCRIPT
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/target_failed.cmake)

if(DOXYGEN_FOUND)
    execute_process(COMMAND "${DOXYGEN_EXECUTABLE}" "--version"
                    OUTPUT_VARIABLE DOXYGEN_VERSION
                    OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(DOXYGEN_VERSION VERSION_LESS MIN_DOXYGEN_VERSION)
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
        add_custom_target(doc_${PROJECT_NAME}_and_sync
            COMMAND ${CMAKE_COMMAND} -P ${TARGET_FAILED_SCRIPT})
    else(DOXYGEN_VERSION VERSION_LESS MIN_DOXYGEN_VERSION)
        # doc_fl target
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in
            ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
        add_custom_target(doc_fl
            ${CMAKE_COMMAND} ${CMAKE_CURRENT_SOURCE_DIR}
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen" VERBATIM)

        # doc_${PROJECT_NAME}_and_sync target
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/sync_doc.cmake.in
            ${CMAKE_CURRENT_BINARY_DIR}/cmake/sync_doc.cmake @ONLY)
        add_custom_target(doc_${PROJECT_NAME}_and_sync
            #${CMAKE_COMMAND} ${CMAKE_CURRENT_SOURCE_DIR}
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
            COMMAND ${CMAKE_COMMAND} -P
                    ${CMAKE_CURRENT_BINARY_DIR}/cmake/sync_doc.cmake
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen" VERBATIM)

    endif(DOXYGEN_VERSION VERSION_LESS MIN_DOXYGEN_VERSION)
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
