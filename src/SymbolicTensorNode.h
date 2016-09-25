#ifndef NUMGRIND_SYMBOLICTENSORNODE_H
#define NUMGRIND_SYMBOLICTENSORNODE_H

#include "SymbolicGraphNode.h"
#include "GNTensorOutput.h"

class SymbolicTensorNode : public SymbolicGraphNode
{
public:
    SymbolicTensorNode(GraphManagerAbstract *manager, GNTensorOutput *graphNode);
    virtual ~SymbolicTensorNode() override;

    const Eigen::MatrixXf &value() const { return this->mGraphNode->value(); }

    GNTensorOutput *node() { return this->mGraphNode; }

protected:
    GNTensorOutput *mGraphNode;
};


#endif //NUMGRIND_SYMBOLICTENSORNODE_H
