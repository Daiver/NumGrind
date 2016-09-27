#ifndef NUMGRINDTEST01_NUMGRIND_H
#define NUMGRINDTEST01_NUMGRIND_H

#include "CompGraph/GraphNode.h"
#include "CompGraph/GNScalarFunction.h"
#include "CompGraph/GNTensorOutput.h"
#include "CompGraph/GNScalarOutput.h"

#include "CompGraph/GNScalarVariable.h"
#include "CompGraph/GNScalarConst.h"
#include "CompGraph/GNScalarSum.h"
#include "CompGraph/GNScalarSub.h"
#include "CompGraph/GNScalarMult.h"

#include "CompGraph/GNVectorVariable.h"
#include "CompGraph/GNDotProduct.h"
#include "CompGraph/GNMatrixSum.h"
#include "CompGraph/GNMatrixElementWiseProduct.h"

#include "CompGraph/GNMatrixConstant.h"
#include "CompGraph/GNMatrixVariable.h"
#include "CompGraph/GNMatrixReduceSum.h"
#include "CompGraph/GNMatrixProduct.h"
#include "CompGraph/GNMatrixTranspose.h"
#include "CompGraph/GNMatrixMapUnaryFunction.h"
#include "CompGraph/GNMatrixSub.h"
#include "CompGraph/GNMatrixScalarSum.h"

#include "GraphManager.h"
#include "SymbolicScalarNode.h"
#include "SymbolicScalarPlaceholder.h"
#include "SymbolicScalarNodeOperators.h"
#include "SymbolicTensorNode.h"
#include "SymbolicTensorPlaceholder.h"
#include "SymbolicTensorNodeOperators.h"

namespace NumGrind {

}

#endif //NUMGRINDTEST01_NUMGRIND_H
