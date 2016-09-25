#ifndef NUMGRIND_GRAPHMANAGER_H
#define NUMGRIND_GRAPHMANAGER_H

#include <vector>
#include "Eigen/Core"
#include "GraphManagerAbstract.h"
#include "SymbolicScalarPlaceholder.h"

class GraphManager : public GraphManagerAbstract{
public:
    GraphManager();
    ~GraphManager();

    virtual void addGraphNode(GraphNode *node) override;

    Eigen::VectorXf initializeVariables();
    static Eigen::VectorXf initializeGradient(const Eigen::VectorXf &vars);

    SymbolicScalarPlaceholder variable(const float val = 0.0);

private:
    std::vector<GNScalarVariable *> mScalarVariables;
    std::vector<GraphNode *> mGraphNodes;
};


#endif //NUMGRIND_GRAPHMANAGER_H
