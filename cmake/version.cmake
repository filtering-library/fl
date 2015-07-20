
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
