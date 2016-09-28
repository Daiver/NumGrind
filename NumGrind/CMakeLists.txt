cmake_minimum_required(VERSION 3.0)
#include_directories("../3rdparty")
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(.)
add_subdirectory(CompGraph)
add_subdirectory(SymbolicGraph)
add_subdirectory(solvers)

set(CORE_FILES
        GraphManager.cpp GraphManager.h
        utils.h numgrind.h)

add_library(numgrind ${CORE_FILES})
target_link_libraries(numgrind NumGrindCompGraph NumGrindSolvers NumGrindSymbolicGraph)
#add_executable(NumGrindTest01 main.cpp )
#target_link_libraries(NumGrindTest01 core)
