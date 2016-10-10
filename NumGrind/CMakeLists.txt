cmake_minimum_required(VERSION 3.0)
#include_directories("../3rdparty")
include_directories(.)
include_directories(${EIGEN3_INCLUDE_DIR})

add_subdirectory(CompGraph)
add_subdirectory(SymbolicGraph)
add_subdirectory(Solvers)
add_subdirectory(DeepGrind)

set(CORE_FILES
        numgrind.h
        GraphManager.cpp GraphManager.h
        utils.h utils.cpp)

add_library(numgrind ${CORE_FILES})
target_link_libraries(numgrind NumGrindCompGraph NumGrindSolvers NumGrindSymbolicGraph DeepGrind)
