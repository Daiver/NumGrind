#ifndef NUMGRIND_GRAPHMANAGERABSTRACT_H
#define NUMGRIND_GRAPHMANAGERABSTRACT_H

#include "GraphNode.h"

namespace NumGrind {
    class GraphManagerAbstract {
    public:
        virtual void addGraphNode(CompGraph::GraphNode *node) = 0;

    };
}

#endif //NUMGRIND_GRAPHMANAGERABSTRACT_H
