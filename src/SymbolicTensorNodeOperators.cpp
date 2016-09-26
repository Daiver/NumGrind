#include "SymbolicTensorNodeOperators.h"

SymbolicScalarNode SymbolicTensorNodeOperators::dot(SymbolicTensorNode a, SymbolicTensorNode b) {
    GraphManagerAbstract *m = a.manager();
    assert(b.manager() == m);

    GNDotProduct *node = new GNDotProduct(a.node(), b.node());
    m->addGraphNode(node);

    return SymbolicScalarNode(m, node);
}

SymbolicTensorNode SymbolicTensorNodeOperators::matmult(SymbolicTensorNode a, SymbolicTensorNode b) {
    GraphManagerAbstract *m = a.manager();
    assert(b.manager() == m);

    GNMatrixProduct *node = new GNMatrixProduct(a.node(), b.node());
    m->addGraphNode(node);

    return SymbolicTensorNode(m, node);
}

SymbolicTensorNode SymbolicTensorNodeOperators::operator+(SymbolicTensorNode a, SymbolicTensorNode b)
{
    GraphManagerAbstract *m = a.manager();
    assert(b.manager() == m);

    GNMatrixSum *node = new GNMatrixSum(a.node(), b.node());
    m->addGraphNode(node);

    return SymbolicTensorNode(m, node);
}

SymbolicTensorNode SymbolicTensorNodeOperators::operator-(SymbolicTensorNode a, SymbolicTensorNode b)
{
    GraphManagerAbstract *m = a.manager();
    assert(b.manager() == m);

    GNMatrixSub *node = new GNMatrixSub(a.node(), b.node());
    m->addGraphNode(node);

    return SymbolicTensorNode(m, node);
}

SymbolicTensorNode SymbolicTensorNodeOperators::operator*(SymbolicTensorNode a, SymbolicTensorNode b)
{
    GraphManagerAbstract *m = a.manager();
    assert(b.manager() == m);

    GNMatrixElementWiseProduct *node = new GNMatrixElementWiseProduct(a.node(), b.node());
    m->addGraphNode(node);

    return SymbolicTensorNode(m, node);
}

SymbolicTensorNode SymbolicTensorNodeOperators::operator+(SymbolicTensorNode a, SymbolicScalarNode b)
{
    GraphManagerAbstract *m = a.manager();
    assert(b.manager() == m);
    auto node = new GNMatrixScalarSum(a.node(), b.node());
    m->addGraphNode(node);
    return SymbolicTensorNode(m, node);
}