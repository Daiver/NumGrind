#ifndef NUMGRIND_GRAPHMANAGER_H
#define NUMGRIND_GRAPHMANAGER_H

#include <vector>
#include "Eigen/Core"
#include "GraphManagerAbstract.h"
#include "SymbolicScalarPlaceholder.h"
#include "SymbolicTensorPlaceholder.h"

namespace NumGrind {
    class GraphManager : public GraphManagerAbstract {
    public:
        GraphManager();

        ~GraphManager();

        virtual void addGraphNode(CompGraph::GraphNode *node) override;

        Eigen::VectorXf initializeVariables();

        static Eigen::VectorXf initializeGradient(const Eigen::VectorXf &vars);

        SymbolicScalarPlaceholder variable(const float val = 0.0);

        SymbolicTensorPlaceholder variable(const int nRows, const int nCols, const float val = 0.0);

        SymbolicTensorPlaceholder variable(const Eigen::MatrixXf &value);

        SymbolicTensorNode constant(const Eigen::MatrixXf &value);

    protected:
        int nVarsForMatrices() const;

    private:
        std::vector<CompGraph::GNScalarVariable *> mScalarVariables;
        std::vector<CompGraph::GNMatrixVariable *> mMatrixVariables;
        std::vector<CompGraph::GraphNode *> mGraphNodes;
    };

}
#endif //NUMGRIND_GRAPHMANAGER_H
