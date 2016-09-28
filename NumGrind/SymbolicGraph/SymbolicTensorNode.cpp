#include "SymbolicTensorNode.h"

using namespace NumGrind::SymbolicGraph;

SymbolicTensorNode::SymbolicTensorNode(GraphManagerAbstract *manager, CompGraph::GNTensorOutput *graphNode): SymbolicGraphNode (manager), mGraphNode(graphNode)
{

}

SymbolicTensorNode::~SymbolicTensorNode() {

}
