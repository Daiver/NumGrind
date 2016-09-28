cmake_minimum_required(VERSION 3.0)

include_directories(${EIGEN3_INCLUDE_DIR})

set(NUMGRIND_SYMBOLICGRAPH_FILES
    tmp.cpp)

add_library(NumGrindSymbolicGraph ${NUMGRIND_SYMBOLICGRAPH_FILES})
target_link_libraries(NumGrindSymbolicGraph NumGrindCompGraph)