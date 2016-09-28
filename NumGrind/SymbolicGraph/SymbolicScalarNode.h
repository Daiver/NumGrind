#ifndef NUMGRIND_SCALARVARIABLE_H
#define NUMGRIND_SCALARVARIABLE_H

#include "SymbolicGraphNode.h"
#include "CompGraph/CGScalarOutput.h"

namespace NumGrind {
    namespace SymbolicGraph {
        class SymbolicScalarNode : public SymbolicGraphNode {
        public:
            SymbolicScalarNode(SymbolicGraphManagerAbstract *manager, CompGraph::CGScalarOutput *graphNode);

            float value() const { return mGraphNode->value(); }

            CompGraph::CGScalarOutput *node();

        protected:
            CompGraph::CGScalarOutput *mGraphNode;
        };
    }
}

#endif //NUMGRIND_SCALARVARIABLE_H
