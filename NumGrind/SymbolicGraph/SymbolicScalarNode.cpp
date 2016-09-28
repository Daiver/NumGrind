//
// Created by daiver on 25.09.16.
//

#include "SymbolicScalarNode.h"

using namespace NumGrind;
using namespace NumGrind::SymbolicGraph;

SymbolicScalarNode::SymbolicScalarNode(SymbolicGraphManagerAbstract *manager, CompGraph::CGScalarOutput *graphNode) : SymbolicGraphNode(manager), mGraphNode(graphNode)
{

}

CompGraph::CGScalarOutput *SymbolicScalarNode::node() {
    return mGraphNode;
}
