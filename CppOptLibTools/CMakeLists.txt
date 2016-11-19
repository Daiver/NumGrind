cmake_minimum_required(VERSION 3.0)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories("../3rdparty/CppNumericalSolvers/include/")
include_directories(../NumGrind)

set(CppOptLibTools_FILES
        NumGrindProblem.h)

#set(CORE_FILES
#        numgrind.h
#        GraphManager.cpp GraphManager.h
#        )

#add_library(numgrind ${CORE_FILES})
#target_link_libraries(numgrind
#    NumGrindCompGraph NumGrindSolvers NumGrindSymbolicGraph DeepGrind Utils)
