cmake_minimum_required(VERSION 3.0)

include_directories(${EIGEN3_INCLUDE_DIR})

set(NUMGRIND_COMPGRAPH_FILES tmp.cpp)
add_library(NumGrindCompGraph ${NUMGRIND_COMPGRAPH_FILES})