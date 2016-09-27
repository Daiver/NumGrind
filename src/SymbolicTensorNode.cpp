#include "SymbolicTensorNode.h"

using namespace NumGrind;

SymbolicTensorNode::SymbolicTensorNode(GraphManagerAbstract *manager, CompGraph::GNTensorOutput *graphNode): SymbolicGraphNode (manager), mGraphNode(graphNode)
{

}

SymbolicTensorNode::~SymbolicTensorNode() {

}
