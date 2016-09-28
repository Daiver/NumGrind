//
// Created by daiver on 25.09.16.
//

#include "SymbolicGraphNode.h"

using namespace NumGrind;
using namespace NumGrind::SymbolicGraph;

SymbolicGraphNode::SymbolicGraphNode(SymbolicGraphManagerAbstract *manager): mManager(manager)
{

}

SymbolicGraphNode::~SymbolicGraphNode() {

}

SymbolicGraphManagerAbstract *SymbolicGraphNode::manager() {
    return this->mManager;
}
