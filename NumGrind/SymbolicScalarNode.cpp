//
// Created by daiver on 25.09.16.
//

#include "SymbolicScalarNode.h"

using namespace NumGrind;
using namespace NumGrind::SymbolicGraph;

SymbolicScalarNode::SymbolicScalarNode(GraphManagerAbstract *manager, CompGraph::GNScalarOutput *graphNode) : SymbolicGraphNode(manager), mGraphNode(graphNode)
{

}

CompGraph::GNScalarOutput *SymbolicScalarNode::node() {
    return mGraphNode;
}
