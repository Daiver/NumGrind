#include "SymbolicTensorNode.h"


SymbolicTensorNode::SymbolicTensorNode(GraphManagerAbstract *manager, GNTensorOutput *graphNode): SymbolicGraphNode (manager), mGraphNode(graphNode)
{

}


SymbolicTensorNode::~SymbolicTensorNode() {

}
