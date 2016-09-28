#ifndef NUMGRIND_SYMBOLICTENSORNODE_H
#define NUMGRIND_SYMBOLICTENSORNODE_H

#include "SymbolicGraphNode.h"
#include "CompGraph/CGTensorOutput.h"

namespace NumGrind {
    namespace SymbolicGraph {
        class SymbolicTensorNode : public SymbolicGraphNode {
        public:
            SymbolicTensorNode(SymbolicGraphManagerAbstract *manager, CompGraph::CGTensorOutput *graphNode);

            virtual ~SymbolicTensorNode() override;

            const Eigen::MatrixXf &value() const { return this->mGraphNode->value(); }

            CompGraph::CGTensorOutput *node() { return this->mGraphNode; }

        protected:
            CompGraph::CGTensorOutput *mGraphNode;
        };
    }
}

#endif //NUMGRIND_SYMBOLICTENSORNODE_H
