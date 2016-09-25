#ifndef NUMGRIND_SCALARVARIABLE_H
#define NUMGRIND_SCALARVARIABLE_H

#include "SymbolicGraphNode.h"
#include "GNScalarOutput.h"

class SymbolicScalarNode : public SymbolicGraphNode {
public:
    SymbolicScalarNode(GraphManagerAbstract *manager, GNScalarOutput *graphNode);

    float value() const { return mGraphNode->value(); }

    GNScalarOutput *node();

protected:
    GNScalarOutput *mGraphNode;
};


#endif //NUMGRIND_SCALARVARIABLE_H
