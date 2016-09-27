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

namespace NumGrind {

    namespace SymbolicNodeOps {

        SymbolicScalarNode dot(SymbolicTensorNode a, SymbolicTensorNode b);

        SymbolicScalarNode reduceSum(SymbolicTensorNode a);

        SymbolicTensorNode matmult(SymbolicTensorNode a, SymbolicTensorNode b);

        template<float Func(float), float Der(float)>
        SymbolicTensorNode apply(SymbolicTensorNode a);

        SymbolicTensorNode operator+(SymbolicTensorNode a, SymbolicTensorNode b);

        SymbolicTensorNode operator-(SymbolicTensorNode a, SymbolicTensorNode b);

        SymbolicTensorNode operator*(SymbolicTensorNode a, SymbolicTensorNode b);

        SymbolicTensorNode operator+(SymbolicTensorNode a, SymbolicScalarNode b);
    };

}

















//IMPLEMENTATIONS
template <float Func(float), float Der(float)>
NumGrind::SymbolicTensorNode NumGrind::SymbolicNodeOps::apply(NumGrind::SymbolicTensorNode a)
{
    NumGrind::GraphManagerAbstract *m = a.manager();
    auto node = new NumGrind::CompGraph::GNMatrixMapUnaryFunction<float, Func, Der>(a.node());
    m->addGraphNode(node);
    return NumGrind::SymbolicTensorNode(m, node);
};


#endif //NUMGRIND_SYMBOLICTENSORNODEOPERATORS_H
