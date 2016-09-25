cmake_minimum_required(VERSION 3.0)

include_directories("../3rdparty")

set(CORE_FILES
        utils.h numgrind.h
        GraphNode.cpp GraphNode.h GNScalarFunction.cpp GNScalarFunction.h GNScalarOutput.cpp GNScalarOutput.h GNTensorOutput.cpp GNTensorOutput.h
        GNScalarVariable.cpp GNScalarVariable.h GNScalarConst.cpp GNScalarConst.h
        GNScalarSum.cpp GNScalarSum.h GNScalarMult.cpp GNScalarMult.h
        GNScalarSub.cpp GNScalarSub.h
        GNVectorVariable.cpp GNVectorVariable.h
        GNDotProduct.cpp GNDotProduct.h GNMatrixSum.cpp GNMatrixSum.h GNVectorElementWiseProduct.cpp GNVectorElementWiseProduct.h GNMatrixVariable.cpp GNMatrixVariable.h
        GNMatrixProduct.cpp GNMatrixProduct.h GNMatrixElementsSum.cpp GNMatrixElementsSum.h GNMatrixTranspose.cpp GNMatrixTranspose.h GNMatrixMapUnaryFunction.h
        GNMatrixConstant.cpp GNMatrixConstant.h GNMatrixSub.cpp GNMatrixSub.h GNMatrixScalarSum.cpp GNMatrixScalarSum.h
        GradientDescentSolver.cpp GradientDescentSolver.h
        GraphManager.cpp GraphManager.h
        GraphManagerAbstract.cpp GraphManagerAbstract.h
        SymbolicGraphNode.cpp SymbolicGraphNode.h
        SymbolicScalarNode.cpp SymbolicScalarNode.h
        SymbolicScalarNodeOperators.cpp SymbolicScalarNodeOperators.h
        SymbolicScalarPlaceholder.cpp SymbolicScalarPlaceholder.h
        )

add_library(core ${CORE_FILES})
add_executable(NumGrindTest01 main.cpp )
target_link_libraries(NumGrindTest01 core)
