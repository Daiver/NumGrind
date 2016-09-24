cmake_minimum_required(VERSION 3.0)
set(SOURCE_FILES main.cpp GraphNode.cpp GraphNode.h GNScalarFunction.cpp GNScalarFunction.h GNScalarOutput.cpp GNScalarOutput.h GNTensorOutput.cpp GNTensorOutput.h utils.h numgrind.h)

add_library(core ${SOURCE_FILES} GNScalarVariable.cpp GNScalarVariable.h GNScalarConst.cpp GNScalarConst.h GNScalarSum.cpp GNScalarSum.h GNScalarMult.cpp GNScalarMult.h GNVectorVariable.cpp GNVectorVariable.h GNDotProduct.cpp GNDotProduct.h GNVectorElementWiseSum.cpp GNVectorElementWiseSum.h GNVectorElementWiseProduct.cpp GNVectorElementWiseProduct.h GNMatrixVariable.cpp GNMatrixVariable.h GNMatrixProduct.cpp GNMatrixProduct.h GNMatrixElementsSum.cpp GNMatrixElementsSum.h GNMatrixTranspose.cpp GNMatrixTranspose.h GNMatrixMapUnaryFunction.h GNMatrixConstant.cpp GNMatrixConstant.h)
add_executable(NumGrindTest01 ${SOURCE_FILES})
target_link_libraries(NumGrindTest01 core)