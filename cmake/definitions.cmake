##########################################################
# adds the definitions
#
# fl_USE_RANDOM_SEED
# fl_USE_FLOAT  OR  fl_USE_DOUBLE  OR  fl_USE_LONG_DOUBLE
##########################################################

if(fl_USE_RANDOM_SEED)
    add_definitions(-Dfl_USE_RANDOM_SEED=1)
endif(fl_USE_RANDOM_SEED)


if(fl_FLOATING_POINT_TYPE STREQUAL "float")
    add_definitions(-Dfl_USE_FLOAT=1)
elseif(fl_FLOATING_POINT_TYPE STREQUAL "double")
    add_definitions(-Dfl_USE_DOUBLE=1)
elseif(fl_FLOATING_POINT_TYPE STREQUAL "long double")
    add_definitions(-Dfl_USE_LONG_DOUBLE=1)
else(fl_FLOATING_POINT_TYPE STREQUAL "float")
    if(fl_FLOATING_POINT_TYPE)
        message(WARNING
            "Unknown floating point type "
            "fl_FLOATING_POINT_TYPE=${fl_FLOATING_POINT_TYPE}. "
            "Falling back to default (double)")
        set(fl_FLOATING_POINT_TYPE "double"
            CACHE STRING "fl::Real floating point type" FORCE)
    endif(fl_FLOATING_POINT_TYPE)

    add_definitions(-Dfl_USE_DOUBLE=1)
endif(fl_FLOATING_POINT_TYPE STREQUAL "float")
