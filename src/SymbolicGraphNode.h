#ifndef NUMGRIND_SYMBOLICGRAPHNODE_H
#define NUMGRIND_SYMBOLICGRAPHNODE_H

#include "GraphManagerAbstract.h"

class SymbolicGraphNode {
public:
    SymbolicGraphNode(GraphManagerAbstract *manager);
    virtual ~SymbolicGraphNode() = 0;

    GraphManagerAbstract *manager();

protected:
    GraphManagerAbstract *mManager;
};


#endif //NUMGRIND_SYMBOLICGRAPHNODE_H
