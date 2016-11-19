#ifndef NUMGRIND_SYMBOLICTENSORNODEOPERATORS_H
#define NUMGRIND_SYMBOLICTENSORNODEOPERATORS_H

#include "SymbolicScalarNode.h"
#include "SymbolicTensorNode.h"

#include "CompGraph/CGMatrixElementWiseProduct.h"
#include "CompGraph/CGDotProduct.h"
#include "CompGraph/CGMatrixConstant.h"
#include "CompGraph/CGMatrixProduct.h"
#include "CompGraph/CGMatrixSum.h"
#include "CompGraph/CGMatrixSub.h"
#include "CompGraph/CGMatrixScalarSum.h"
#include "CompGraph/CGMatrixMapUnaryFunction.h"
#include "CompGraph/CGMatrixReduceSum.h"

namespace NumGrind {

    namespace SymbolicGraph {

        SymbolicScalarNode dot(SymbolicTensorNode a, SymbolicTensorNode b);

        SymbolicScalarNode reduceSum(SymbolicTensorNode a);

        SymbolicTensorNode matmult(SymbolicTensorNode a, SymbolicTensorNode b);

        SymbolicScalarNode sumOfSquares(SymbolicTensorNode a);

        template<float Func(const float), float Der(const float)>
        SymbolicTensorNode apply(SymbolicTensorNode a);

        SymbolicTensorNode operator+(SymbolicTensorNode a, SymbolicTensorNode b);

        SymbolicTensorNode operator-(SymbolicTensorNode a, SymbolicTensorNode b);

        SymbolicTensorNode operator*(SymbolicTensorNode a, SymbolicTensorNode b);

        SymbolicTensorNode operator+(SymbolicTensorNode a, SymbolicScalarNode b);
    };

}

















//IMPLEMENTATIONS
template <float Func(const float), float Der(const float)>
NumGrind::SymbolicGraph::SymbolicTensorNode NumGrind::SymbolicGraph::apply(NumGrind::SymbolicGraph::SymbolicTensorNode a)
{
    NumGrind::SymbolicGraph::SymbolicGraphManagerAbstract *m = a.manager();
    auto node = new NumGrind::CompGraph::CGMatrixMapUnaryFunction<float, Func, Der>(a.node());
    m->addGraphNode(node);
    return NumGrind::SymbolicGraph::SymbolicTensorNode(m, node);
};


#endif //NUMGRIND_SYMBOLICTENSORNODEOPERATORS_H
