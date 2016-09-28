#ifndef NUMGRINDTEST01_NUMGRIND_H
#define NUMGRINDTEST01_NUMGRIND_H

#include "CompGraph/CompGraphNode.h"
#include "CompGraph/CGScalarFunction.h"
#include "CompGraph/CGTensorOutput.h"
#include "CompGraph/CGScalarOutput.h"

#include "CompGraph/CGScalarVariable.h"
#include "CompGraph/CGScalarConst.h"
#include "CompGraph/CGScalarSum.h"
#include "CompGraph/CGScalarSub.h"
#include "CompGraph/CGScalarMult.h"

#include "CompGraph/CGVectorVariable.h"
#include "CompGraph/CGDotProduct.h"
#include "CompGraph/CGMatrixSum.h"
#include "CompGraph/CGMatrixElementWiseProduct.h"

#include "CompGraph/CGMatrixConstant.h"
#include "CompGraph/CGMatrixVariable.h"
#include "CompGraph/CGMatrixReduceSum.h"
#include "CompGraph/CGMatrixProduct.h"
#include "CompGraph/CGMatrixTranspose.h"
#include "CompGraph/CGMatrixMapUnaryFunction.h"
#include "CompGraph/CGMatrixSub.h"
#include "CompGraph/CGMatrixScalarSum.h"

#include "GraphManager.h"
#include "SymbolicGraph/SymbolicScalarNode.h"
#include "SymbolicGraph/SymbolicScalarPlaceholder.h"
#include "SymbolicGraph/SymbolicScalarNodeOperators.h"
#include "SymbolicGraph/SymbolicTensorNode.h"
#include "SymbolicGraph/SymbolicTensorPlaceholder.h"
#include "SymbolicGraph/SymbolicTensorNodeOperators.h"

namespace NumGrind {

}

#endif //NUMGRINDTEST01_NUMGRIND_H
