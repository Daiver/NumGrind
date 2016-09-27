#ifndef NUMGRIND_SCALARVARIABLE_H
#define NUMGRIND_SCALARVARIABLE_H

#include "SymbolicGraphNode.h"
#include "GNScalarOutput.h"

namespace NumGrind {
    class SymbolicScalarNode : public SymbolicGraphNode {
    public:
        SymbolicScalarNode(GraphManagerAbstract *manager, CompGraph::GNScalarOutput *graphNode);

        float value() const { return mGraphNode->value(); }

        CompGraph::GNScalarOutput *node();

    protected:
        CompGraph::GNScalarOutput *mGraphNode;
    };
}

#endif //NUMGRIND_SCALARVARIABLE_H
