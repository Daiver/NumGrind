#ifndef NUMGRIND_GRAPHMANAGER_H
#define NUMGRIND_GRAPHMANAGER_H

#include <vector>
#include "Eigen/Core"
#include "GraphManagerAbstract.h"
#include "SymbolicScalarPlaceholder.h"
#include "SymbolicTensorPlaceholder.h"


class GraphManager : public GraphManagerAbstract{
public:
    GraphManager();
    ~GraphManager();

    virtual void addGraphNode(GraphNode *node) override;

    Eigen::VectorXf initializeVariables();
    static Eigen::VectorXf initializeGradient(const Eigen::VectorXf &vars);

    SymbolicScalarPlaceholder variable(const float val = 0.0);
    SymbolicTensorPlaceholder variable(const int nRows, const int nCols, const float val = 0.0);
    SymbolicTensorPlaceholder variable(const Eigen::MatrixXf &value);

    SymbolicTensorNode constant(const Eigen::MatrixXf &value);

protected:
    int nVarsForMatrices() const;

private:
    std::vector<GNScalarVariable *> mScalarVariables;
    std::vector<GNMatrixVariable *> mMatrixVariables;
    std::vector<GraphNode *> mGraphNodes;
};


#endif //NUMGRIND_GRAPHMANAGER_H
