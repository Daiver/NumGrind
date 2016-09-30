#include "SymbolicTensorNodeOperators.h"

#include "CompGraph/CGSumOfSquares.h"

using namespace NumGrind;
using namespace NumGrind::CompGraph;
using namespace NumGrind::SymbolicGraph;

SymbolicScalarNode SymbolicGraph::dot(SymbolicTensorNode a, SymbolicTensorNode b) {
    SymbolicGraphManagerAbstract *m = a.manager();
    assert(b.manager() == m);

    CGDotProduct *node = new CGDotProduct(a.node(), b.node());
    m->addGraphNode(node);

    return SymbolicScalarNode(m, node);
}

SymbolicTensorNode SymbolicGraph::matmult(SymbolicTensorNode a, SymbolicTensorNode b) {
    SymbolicGraphManagerAbstract *m = a.manager();
    assert(b.manager() == m);

    CGMatrixProduct *node = new CGMatrixProduct(a.node(), b.node());
    m->addGraphNode(node);

    return SymbolicTensorNode(m, node);
}

SymbolicTensorNode SymbolicGraph::operator+(SymbolicTensorNode a, SymbolicTensorNode b)
{
    SymbolicGraphManagerAbstract *m = a.manager();
    assert(b.manager() == m);

    CGMatrixSum *node = new CGMatrixSum(a.node(), b.node());
    m->addGraphNode(node);

    return SymbolicTensorNode(m, node);
}

SymbolicTensorNode SymbolicGraph::operator-(SymbolicTensorNode a, SymbolicTensorNode b)
{
    SymbolicGraphManagerAbstract *m = a.manager();
    assert(b.manager() == m);

    CGMatrixSub *node = new CGMatrixSub(a.node(), b.node());
    m->addGraphNode(node);

    return SymbolicTensorNode(m, node);
}

SymbolicTensorNode SymbolicGraph::operator*(SymbolicTensorNode a, SymbolicTensorNode b)
{
    SymbolicGraphManagerAbstract *m = a.manager();
    assert(b.manager() == m);

    CGMatrixElementWiseProduct *node = new CGMatrixElementWiseProduct(a.node(), b.node());
    m->addGraphNode(node);

    return SymbolicTensorNode(m, node);
}

SymbolicTensorNode SymbolicGraph::operator+(SymbolicTensorNode a, SymbolicScalarNode b)
{
    SymbolicGraphManagerAbstract *m = a.manager();
    assert(b.manager() == m);
    auto node = new CGMatrixScalarSum(a.node(), b.node());
    m->addGraphNode(node);
    return SymbolicTensorNode(m, node);
}

SymbolicScalarNode SymbolicGraph::reduceSum(SymbolicTensorNode a)
{
    SymbolicGraphManagerAbstract *m = a.manager();
    auto node = new CGMatrixReduceSum(a.node());
    m->addGraphNode(node);
    return SymbolicScalarNode(m, node);
}

SymbolicScalarNode SymbolicGraph::sumOfSquares(SymbolicTensorNode a)
{
    SymbolicGraphManagerAbstract *m = a.manager();
    auto node = new CGSumOfSquares(a.node());
    m->addGraphNode(node);
    return SymbolicScalarNode(m, node);
}