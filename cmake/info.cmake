string(ASCII 27 Esc)

set(fl_USING_CATKIN_STR "no")
set(fl_USING_RANDOM_SEED_STR "no")
set(fl_FLOATING_POINT_TYPE_STR "default")

if(fl_USING_CATKIN)
    set(fl_USING_CATKIN_STR "yes")
endif(fl_USING_CATKIN)

if(fl_USE_RANDOM_SEED)
    set(fl_USING_RANDOM_SEED_STR "yes")
endif(fl_USE_RANDOM_SEED)

if(fl_FLOATING_POINT_TYPE STREQUAL "float")
    set(fl_FLOATING_POINT_TYPE_STR "float")
elseif(fl_FLOATING_POINT_TYPE STREQUAL "double")
    set(fl_FLOATING_POINT_TYPE_STR "double")
elseif(fl_FLOATING_POINT_TYPE STREQUAL "long double")
    set(fl_FLOATING_POINT_TYPE_STR "long double")
else(fl_FLOATING_POINT_TYPE STREQUAL "float")
    set(fl_FLOATING_POINT_TYPE_STR "double")
endif(fl_FLOATING_POINT_TYPE STREQUAL "float")

set(fl_COLOR "${Esc}[3;35m")
set(fl_COLOR_CLEAR "${Esc}[m")

message(STATUS "${fl_COLOR} ================================================= ${fl_COLOR_CLEAR}")
message(STATUS "${fl_COLOR} == ${Esc}[3;34m Filtering Library ${fl_COLOR_CLEAR}")
message(STATUS "${fl_COLOR} == ${Esc}[3;34m Version${Esc}[1;34m ${PROJECT_VERSION} ${fl_COLOR_CLEAR}")
message(STATUS "${fl_COLOR} == ${fl_COLOR_CLEAR}")
message(STATUS "${fl_COLOR} == ${Esc}[1;34m Setup: ${fl_COLOR_CLEAR}")
message(STATUS "${fl_COLOR} == ${fl_COLOR_CLEAR} - Using Catkin: ${fl_USING_CATKIN_STR}")
message(STATUS "${fl_COLOR} == ${fl_COLOR_CLEAR} - Using random seeds: ${fl_USING_RANDOM_SEED_STR}")
message(STATUS "${fl_COLOR} == ${fl_COLOR_CLEAR} - Using fl::Real floating point type: ${fl_FLOATING_POINT_TYPE_STR}")
message(STATUS "${fl_COLOR} ================================================= ${fl_COLOR_CLEAR} ")

