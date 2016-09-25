#include "SymbolicScalarNodeOperators.h"

SymbolicScalarNode  SymbolicScalarNodeOperators::operator+(SymbolicScalarNode &a, SymbolicScalarNode &b) {
    GraphManagerAbstract *manager = a.manager();
    assert(manager == b.manager());
    GNScalarOutput *node = new GNScalarSum(a.node(), b.node());
    manager->addGraphNode(node);
    return SymbolicScalarNode(manager, node);
}

SymbolicScalarNode  SymbolicScalarNodeOperators::operator*(SymbolicScalarNode &a, SymbolicScalarNode &b) {
    GraphManagerAbstract *manager = a.manager();
    assert(manager == b.manager());
    GNScalarOutput *node = new GNScalarMult(a.node(), b.node());
    manager->addGraphNode(node);
    return SymbolicScalarNode(manager, node);
}