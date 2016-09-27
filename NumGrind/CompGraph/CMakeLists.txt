cmake_minimum_required(VERSION 3.0)

include_directories(${EIGEN3_INCLUDE_DIR})

set(NUMGRIND_COMPGRAPH_FILES
        GraphNode.cpp GraphNode.h
        GNScalarFunction.cpp GNScalarFunction.h
        GNScalarOutput.cpp GNScalarOutput.h
        GNTensorOutput.cpp GNTensorOutput.h
        GNScalarVariable.cpp GNScalarVariable.h
        GNScalarConst.cpp GNScalarConst.h
        GNScalarSum.cpp GNScalarSum.h
        GNScalarMult.cpp GNScalarMult.h
        GNScalarSub.cpp GNScalarSub.h
        GNVectorVariable.cpp GNVectorVariable.h
        GNDotProduct.cpp GNDotProduct.h
        GNMatrixSum.cpp GNMatrixSum.h
        GNMatrixElementWiseProduct.cpp GNMatrixElementWiseProduct.h
        GNMatrixVariable.cpp GNMatrixVariable.h
        GNMatrixProduct.cpp GNMatrixProduct.h
        GNMatrixReduceSum.cpp GNMatrixReduceSum.h
        GNMatrixTranspose.cpp GNMatrixTranspose.h
        GNMatrixMapUnaryFunction.h
        GNMatrixConstant.cpp GNMatrixConstant.h
        GNMatrixSub.cpp GNMatrixSub.h
        GNMatrixScalarSum.cpp GNMatrixScalarSum.h)
add_library(NumGrindCompGraph ${NUMGRIND_COMPGRAPH_FILES})