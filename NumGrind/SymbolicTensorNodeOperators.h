#ifndef NUMGRIND_SYMBOLICTENSORNODEOPERATORS_H
#define NUMGRIND_SYMBOLICTENSORNODEOPERATORS_H

#include "SymbolicScalarNode.h"
#include "SymbolicTensorNode.h"

#include "CompGraph/GNMatrixElementWiseProduct.h"
#include "CompGraph/GNDotProduct.h"
#include "CompGraph/GNMatrixConstant.h"
#include "CompGraph/GNMatrixProduct.h"
#include "CompGraph/GNMatrixSum.h"
#include "CompGraph/GNMatrixSub.h"
#include "CompGraph/GNMatrixScalarSum.h"
#include "CompGraph/GNMatrixMapUnaryFunction.h"
#include "CompGraph/GNMatrixReduceSum.h"

namespace NumGrind {

    namespace SymbolicGraph {

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
NumGrind::SymbolicGraph::SymbolicTensorNode NumGrind::SymbolicGraph::apply(NumGrind::SymbolicGraph::SymbolicTensorNode a)
{
    NumGrind::GraphManagerAbstract *m = a.manager();
    auto node = new NumGrind::CompGraph::GNMatrixMapUnaryFunction<float, Func, Der>(a.node());
    m->addGraphNode(node);
    return NumGrind::SymbolicGraph::SymbolicTensorNode(m, node);
};


#endif //NUMGRIND_SYMBOLICTENSORNODEOPERATORS_H
