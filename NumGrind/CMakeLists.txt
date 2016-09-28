cmake_minimum_required(VERSION 3.0)
#include_directories("../3rdparty")
include_directories(${EIGEN3_INCLUDE_DIR})

add_subdirectory(CompGraph)
add_subdirectory(solvers)

set(CORE_FILES
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
target_link_libraries(numgrind NumGrindCompGraph NumGrindSolvers)
#add_executable(NumGrindTest01 main.cpp )
#target_link_libraries(NumGrindTest01 core)
