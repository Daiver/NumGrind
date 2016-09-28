#include "SymbolicScalarNodeOperators.h"

using namespace NumGrind;
using namespace NumGrind::CompGraph;
using namespace NumGrind::SymbolicGraph;

SymbolicScalarNode  SymbolicGraph::operator+(SymbolicScalarNode a, SymbolicScalarNode b) {
    SymbolicGraphManagerAbstract *manager = a.manager();
    assert(manager == b.manager());
    CGScalarOutput *node = new CGScalarSum(a.node(), b.node());
    manager->addGraphNode(node);
    return SymbolicScalarNode(manager, node);
}

SymbolicScalarNode  SymbolicGraph::operator-(SymbolicScalarNode a, SymbolicScalarNode b) {
    SymbolicGraphManagerAbstract *manager = a.manager();
    assert(manager == b.manager());
    CGScalarOutput *node = new CGScalarSub(a.node(), b.node());
    manager->addGraphNode(node);
    return SymbolicScalarNode(manager, node);
}

SymbolicScalarNode  SymbolicGraph::operator*(SymbolicScalarNode a, SymbolicScalarNode b) {
    SymbolicGraphManagerAbstract *manager = a.manager();
    assert(manager == b.manager());
    CGScalarOutput *node = new CGScalarMult(a.node(), b.node());
    manager->addGraphNode(node);
    return SymbolicScalarNode(manager, node);
}

SymbolicScalarNode  SymbolicGraph::operator+(const float a, SymbolicScalarNode b) {
    SymbolicGraphManagerAbstract *manager = b.manager();
    CGScalarOutput *nodeConstant = new CGScalarConst(a);
    manager->addGraphNode(nodeConstant);
    CGScalarOutput *nodeAdd = new CGScalarSum(nodeConstant, b.node());
    manager->addGraphNode(nodeAdd);
    return SymbolicScalarNode(manager, nodeAdd);
}

SymbolicScalarNode  SymbolicGraph::operator-(const float a, SymbolicScalarNode b) {
    SymbolicGraphManagerAbstract *manager = b.manager();
    CGScalarOutput *nodeConstant = new CGScalarConst(a);
    manager->addGraphNode(nodeConstant);
    CGScalarOutput *nodeSub = new CGScalarSub(nodeConstant, b.node());
    manager->addGraphNode(nodeSub);
    return SymbolicScalarNode(manager, nodeSub);
}

SymbolicScalarNode  SymbolicGraph::operator*(const float a, SymbolicScalarNode b) {
    SymbolicGraphManagerAbstract *manager = b.manager();
    CGScalarOutput *nodeConstant = new CGScalarConst(a);
    manager->addGraphNode(nodeConstant);
    CGScalarOutput *nodeMult = new CGScalarMult(nodeConstant, b.node());
    manager->addGraphNode(nodeMult);
    return SymbolicScalarNode(manager, nodeMult);
}

SymbolicScalarNode  SymbolicGraph::operator+(SymbolicScalarNode a, const float b) {
    SymbolicGraphManagerAbstract *manager = a.manager();
    CGScalarOutput *nodeConstant = new CGScalarConst(b);
    manager->addGraphNode(nodeConstant);
    CGScalarOutput *nodeAdd = new CGScalarSum(a.node(), nodeConstant);
    manager->addGraphNode(nodeAdd);
    return SymbolicScalarNode(manager, nodeAdd);
}

SymbolicScalarNode  SymbolicGraph::operator-(SymbolicScalarNode a, const float b) {
    SymbolicGraphManagerAbstract *manager = a.manager();
    CGScalarOutput *nodeConstant = new CGScalarConst(b);
    manager->addGraphNode(nodeConstant);
    CGScalarOutput *nodeSub = new CGScalarSub(a.node(), nodeConstant);
    manager->addGraphNode(nodeSub);
    return SymbolicScalarNode(manager, nodeSub);
}

SymbolicScalarNode  SymbolicGraph::operator*(SymbolicScalarNode a, const float b) {
    SymbolicGraphManagerAbstract *manager = a.manager();
    CGScalarOutput *nodeConstant = new CGScalarConst(b);
    manager->addGraphNode(nodeConstant);
    CGScalarOutput *nodeMult = new CGScalarMult(a.node(), nodeConstant);
    manager->addGraphNode(nodeMult);
    return SymbolicScalarNode(manager, nodeMult);
}


SymbolicScalarNode  SymbolicGraph::operator/(SymbolicScalarNode a, const float b) {
    SymbolicGraphManagerAbstract *manager = a.manager();
    CGScalarOutput *nodeConstant = new CGScalarConst(1.0/b);
    manager->addGraphNode(nodeConstant);
    CGScalarOutput *nodeAdd = new CGScalarMult(a.node(), nodeConstant);
    manager->addGraphNode(nodeAdd);
    return SymbolicScalarNode(manager, nodeAdd);
}