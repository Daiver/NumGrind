#include "SymbolicScalarNodeOperators.h"

SymbolicScalarNode  SymbolicScalarNodeOperators::operator+(SymbolicScalarNode a, SymbolicScalarNode b) {
    GraphManagerAbstract *manager = a.manager();
    assert(manager == b.manager());
    GNScalarOutput *node = new GNScalarSum(a.node(), b.node());
    manager->addGraphNode(node);
    return SymbolicScalarNode(manager, node);
}

SymbolicScalarNode  SymbolicScalarNodeOperators::operator-(SymbolicScalarNode a, SymbolicScalarNode b) {
    GraphManagerAbstract *manager = a.manager();
    assert(manager == b.manager());
    GNScalarOutput *node = new GNScalarSub(a.node(), b.node());
    manager->addGraphNode(node);
    return SymbolicScalarNode(manager, node);
}

SymbolicScalarNode  SymbolicScalarNodeOperators::operator*(SymbolicScalarNode a, SymbolicScalarNode b) {
    GraphManagerAbstract *manager = a.manager();
    assert(manager == b.manager());
    GNScalarOutput *node = new GNScalarMult(a.node(), b.node());
    manager->addGraphNode(node);
    return SymbolicScalarNode(manager, node);
}

SymbolicScalarNode  SymbolicScalarNodeOperators::operator+(const float a, SymbolicScalarNode b) {
    GraphManagerAbstract *manager = b.manager();
    GNScalarOutput *nodeConstant = new GNScalarConst(a);
    manager->addGraphNode(nodeConstant);
    GNScalarOutput *nodeAdd = new GNScalarSum(nodeConstant, b.node());
    manager->addGraphNode(nodeAdd);
    return SymbolicScalarNode(manager, nodeAdd);
}

SymbolicScalarNode  SymbolicScalarNodeOperators::operator-(const float a, SymbolicScalarNode b) {
    GraphManagerAbstract *manager = b.manager();
    GNScalarOutput *nodeConstant = new GNScalarConst(a);
    manager->addGraphNode(nodeConstant);
    GNScalarOutput *nodeSub = new GNScalarSub(nodeConstant, b.node());
    manager->addGraphNode(nodeSub);
    return SymbolicScalarNode(manager, nodeSub);
}

SymbolicScalarNode  SymbolicScalarNodeOperators::operator*(const float a, SymbolicScalarNode b) {
    GraphManagerAbstract *manager = b.manager();
    GNScalarOutput *nodeConstant = new GNScalarConst(a);
    manager->addGraphNode(nodeConstant);
    GNScalarOutput *nodeMult = new GNScalarMult(nodeConstant, b.node());
    manager->addGraphNode(nodeMult);
    return SymbolicScalarNode(manager, nodeMult);
}

SymbolicScalarNode  SymbolicScalarNodeOperators::operator+(SymbolicScalarNode a, const float b) {
    GraphManagerAbstract *manager = a.manager();
    GNScalarOutput *nodeConstant = new GNScalarConst(b);
    manager->addGraphNode(nodeConstant);
    GNScalarOutput *nodeAdd = new GNScalarSum(a.node(), nodeConstant);
    manager->addGraphNode(nodeAdd);
    return SymbolicScalarNode(manager, nodeAdd);
}

SymbolicScalarNode  SymbolicScalarNodeOperators::operator-(SymbolicScalarNode a, const float b) {
    GraphManagerAbstract *manager = a.manager();
    GNScalarOutput *nodeConstant = new GNScalarConst(b);
    manager->addGraphNode(nodeConstant);
    GNScalarOutput *nodeSub = new GNScalarSub(a.node(), nodeConstant);
    manager->addGraphNode(nodeSub);
    return SymbolicScalarNode(manager, nodeSub);
}

SymbolicScalarNode  SymbolicScalarNodeOperators::operator*(SymbolicScalarNode a, const float b) {
    GraphManagerAbstract *manager = a.manager();
    GNScalarOutput *nodeConstant = new GNScalarConst(b);
    manager->addGraphNode(nodeConstant);
    GNScalarOutput *nodeMult = new GNScalarMult(a.node(), nodeConstant);
    manager->addGraphNode(nodeMult);
    return SymbolicScalarNode(manager, nodeMult);
}


SymbolicScalarNode  SymbolicScalarNodeOperators::operator/(SymbolicScalarNode a, const float b) {
    GraphManagerAbstract *manager = a.manager();
    GNScalarOutput *nodeConstant = new GNScalarConst(1.0/b);
    manager->addGraphNode(nodeConstant);
    GNScalarOutput *nodeAdd = new GNScalarMult(a.node(), nodeConstant);
    manager->addGraphNode(nodeAdd);
    return SymbolicScalarNode(manager, nodeAdd);
}