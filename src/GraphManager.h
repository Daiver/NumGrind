#ifndef NUMGRIND_GRAPHMANAGER_H
#define NUMGRIND_GRAPHMANAGER_H

#include <vector>
#include "Eigen/Core"
#include "GraphManagerAbstract.h"

class GraphManager : public GraphManagerAbstract{
public:
    GraphManager();
    ~GraphManager();

    virtual void addGraphNode(GraphNode *node) override;

private:
    std::vector<GraphNode *> mGraphNodes;
};


#endif //NUMGRIND_GRAPHMANAGER_H
