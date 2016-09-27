#ifndef NUMGRIND_SYMBOLICTENSORNODE_H
#define NUMGRIND_SYMBOLICTENSORNODE_H

#include "SymbolicGraphNode.h"
#include "GNTensorOutput.h"

namespace NumGrind {
    class SymbolicTensorNode : public SymbolicGraphNode {
    public:
        SymbolicTensorNode(GraphManagerAbstract *manager, CompGraph::GNTensorOutput *graphNode);

        virtual ~SymbolicTensorNode() override;

        const Eigen::MatrixXf &value() const { return this->mGraphNode->value(); }

        CompGraph::GNTensorOutput *node() { return this->mGraphNode; }

    protected:
        CompGraph::GNTensorOutput *mGraphNode;
    };
}

#endif //NUMGRIND_SYMBOLICTENSORNODE_H
