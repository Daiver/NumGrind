#include "SymbolicTensorNode.h"

using namespace NumGrind::SymbolicGraph;

SymbolicTensorNode::SymbolicTensorNode(SymbolicGraphManagerAbstract *manager, CompGraph::CGTensorOutput *graphNode): SymbolicGraphNode (manager), mGraphNode(graphNode)
{

}

SymbolicTensorNode::~SymbolicTensorNode() {

}
