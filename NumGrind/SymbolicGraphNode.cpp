//
// Created by daiver on 25.09.16.
//

#include "SymbolicGraphNode.h"

using namespace NumGrind;

SymbolicGraphNode::SymbolicGraphNode(GraphManagerAbstract *manager): mManager(manager)
{

}

SymbolicGraphNode::~SymbolicGraphNode() {

}

GraphManagerAbstract *SymbolicGraphNode::manager() {
    return this->mManager;
}