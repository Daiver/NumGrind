cmake_minimum_required(VERSION 3.0)

include_directories(${EIGEN3_INCLUDE_DIR})

set(NUMGRIND_COMPGRAPH_FILES
        CompGraphNode.cpp CompGraphNode.h
        CGScalarFunction.cpp CGScalarFunction.h
        CGScalarOutput.cpp CGScalarOutput.h
        CGTensorOutput.cpp CGTensorOutput.h
        CGScalarVariable.cpp CGScalarVariable.h
        CGScalarConst.cpp CGScalarConst.h
        CGScalarSum.cpp CGScalarSum.h
        CGScalarMult.cpp CGScalarMult.h
        CGScalarSub.cpp CGScalarSub.h
        CGVectorVariable.cpp CGVectorVariable.h
        CGDotProduct.cpp CGDotProduct.h
        CGMatrixSum.cpp CGMatrixSum.h
        CGMatrixElementWiseProduct.cpp CGMatrixElementWiseProduct.h
        CGMatrixVariable.cpp CGMatrixVariable.h
        CGMatrixProduct.cpp CGMatrixProduct.h
        CGMatrixReduceSum.cpp CGMatrixReduceSum.h
        CGMatrixTranspose.cpp CGMatrixTranspose.h
        CGMatrixMapUnaryFunction.h
        CGMatrixConstant.cpp CGMatrixConstant.h
        CGMatrixSub.cpp CGMatrixSub.h
        CGMatrixScalarSum.cpp CGMatrixScalarSum.h
        CGSumOfSquares.cpp CGSumOfSquares.h)

add_library(NumGrindCompGraph ${NUMGRIND_COMPGRAPH_FILES})