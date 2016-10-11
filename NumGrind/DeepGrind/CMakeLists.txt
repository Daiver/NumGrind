cmake_minimum_required(VERSION 3.0)

include_directories(${EIGEN3_INCLUDE_DIR})

set(NUMGRIND_DEEPGRIND_FILES
        ActivationFunctions.h ActivationFunctions.cpp)
add_library(DeepGrind ${NUMGRIND_DEEPGRIND_FILES})
target_link_libraries(DeepGrind NumGrindSymbolicGraph NumGrindCompGraph)
