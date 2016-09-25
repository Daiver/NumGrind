//
// Created by daiver on 25.09.16.
//

#include "SymbolicScalarNode.h"

SymbolicScalarNode::SymbolicScalarNode(GraphManagerAbstract *manager, GNScalarOutput *graphNode) : SymbolicGraphNode(manager), mGraphNode(graphNode)
{

}

GNScalarOutput *SymbolicScalarNode::node() {
    return mGraphNode;
}
