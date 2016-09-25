#ifndef NUMGRIND_GRAPHMANAGERABSTRACT_H
#define NUMGRIND_GRAPHMANAGERABSTRACT_H

#include "GraphNode.h"

class GraphManagerAbstract {
public:
    virtual void addGraphNode(GraphNode *node) = 0;

};


#endif //NUMGRIND_GRAPHMANAGERABSTRACT_H
