cmake_minimum_required(VERSION 3.0)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(../)

set(NUMGRIND_SYMBOLICGRAPH_FILES
        SymbolicGraphNode.cpp SymbolicGraphNode.h
        SymbolicScalarNode.cpp SymbolicScalarNode.h
        SymbolicScalarNodeOperators.cpp SymbolicScalarNodeOperators.h
        SymbolicScalarVariable.cpp SymbolicScalarVariable.h
        SymbolicTensorNode.cpp SymbolicTensorNode.h
        SymbolicTensorVariable.cpp SymbolicTensorVariable.h
        SymbolicTensorNodeOperators.cpp SymbolicTensorNodeOperators.h
        SymbolicGraphManagerAbstract.cpp SymbolicGraphManagerAbstract.h
        SymbolicTensorConstant.cpp SymbolicTensorConstant.h
        )

add_library(NumGrindSymbolicGraph ${NUMGRIND_SYMBOLICGRAPH_FILES})
target_link_libraries(NumGrindSymbolicGraph NumGrindCompGraph)