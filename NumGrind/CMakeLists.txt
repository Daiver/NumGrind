cmake_minimum_required(VERSION 3.0)
#include_directories("../3rdparty")
include_directories(${EIGEN3_INCLUDE_DIR})

add_subdirectory(CompGraph)

set(CORE_FILES
        GradientDescentSolver.cpp GradientDescentSolver.h
        GraphManager.cpp GraphManager.h
        GraphManagerAbstract.cpp GraphManagerAbstract.h
        SymbolicGraphNode.cpp SymbolicGraphNode.h
        SymbolicScalarNode.cpp SymbolicScalarNode.h
        SymbolicScalarNodeOperators.cpp SymbolicScalarNodeOperators.h
        SymbolicScalarPlaceholder.cpp SymbolicScalarPlaceholder.h
        SymbolicTensorNode.cpp SymbolicTensorNode.h
        SymbolicTensorPlaceholder.cpp SymbolicTensorPlaceholder.h
        SymbolicTensorNodeOperators.cpp SymbolicTensorNodeOperators.h
        utils.h numgrind.h)

add_library(numgrind ${CORE_FILES})
target_link_libraries(numgrind NumGrindCompGraph)
#add_executable(NumGrindTest01 main.cpp )
#target_link_libraries(NumGrindTest01 core)
