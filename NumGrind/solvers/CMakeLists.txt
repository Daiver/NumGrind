cmake_minimum_required(VERSION 3.0)

include_directories(${EIGEN3_INCLUDE_DIR})

set(NUMGRIND_SOLVERS_FILES
        GradientDescentSolver.cpp GradientDescentSolver.h
        )
add_library(NumGrindSolvers ${NUMGRIND_SOLVERS_FILES})