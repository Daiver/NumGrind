#ifndef NUMGRIND_SYMBOLICTENSORNODEOPERATORS_H
#define NUMGRIND_SYMBOLICTENSORNODEOPERATORS_H

#include "SymbolicScalarNode.h"
#include "SymbolicTensorNode.h"

#include "GNMatrixElementWiseProduct.h"
#include "GNDotProduct.h"
#include "GNMatrixConstant.h"
#include "GNMatrixProduct.h"
#include "GNMatrixSum.h"
#include "GNMatrixSub.h"
#include "GNMatrixScalarSum.h"
#include "GNMatrixMapUnaryFunction.h"
#include "GNMatrixReduceSum.h"


namespace SymbolicTensorNodeOperators {

    SymbolicScalarNode dot(SymbolicTensorNode a, SymbolicTensorNode b);
    SymbolicTensorNode matmult(SymbolicTensorNode a, SymbolicTensorNode b);

    SymbolicTensorNode operator+(SymbolicTensorNode a, SymbolicTensorNode b);
    SymbolicTensorNode operator-(SymbolicTensorNode a, SymbolicTensorNode b);
    SymbolicTensorNode operator*(SymbolicTensorNode a, SymbolicTensorNode b);

};


#endif //NUMGRIND_SYMBOLICTENSORNODEOPERATORS_H
