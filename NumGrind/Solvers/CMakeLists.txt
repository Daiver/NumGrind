cmake_minimum_required(VERSION 3.0)

include_directories(${EIGEN3_INCLUDE_DIR})

set(NUMGRIND_SOLVERS_FILES
        SolverSettings.cpp SolverSettings.h
        GradientDescentSolver.cpp GradientDescentSolver.h
        SGDSolver.cpp SGDSolver.h
        numericalgradient.cpp numericalgradient.h
        checkgradient.cpp checkgradient.h SGDWithMomentumSolver.cpp SGDWithMomentumSolver.h)
add_library(NumGrindSolvers ${NUMGRIND_SOLVERS_FILES})