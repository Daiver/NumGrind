#ifndef NUMGRIND_GRAPHMANAGER_H
#define NUMGRIND_GRAPHMANAGER_H

#include <vector>
#include <functional>
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

        SymbolicGraph::SymbolicScalarPlaceholder variable(const float val = 0.0);

        SymbolicGraph::SymbolicTensorPlaceholder variable(const int nRows, const int nCols, const float val = 0.0);

        SymbolicGraph::SymbolicTensorPlaceholder variable(const Eigen::MatrixXf &value);

        SymbolicGraph::SymbolicTensorNode constant(const Eigen::MatrixXf &value);

        static std::function<float(const Eigen::VectorXf&)> funcFromNode(
                SymbolicGraph::SymbolicScalarNode *func)
        {
            return [=](const Eigen::VectorXf &vars){
                func->node()->forwardPass(vars);
                return func->node()->value();
            };
        }

        static std::function<void(const Eigen::VectorXf&, Eigen::VectorXf &)> gradFromNode(
                SymbolicGraph::SymbolicScalarNode *func)
        {
			return [=](const Eigen::VectorXf &vars, Eigen::VectorXf &grad) {
					 func->node()->forwardPass(vars);
					 func->node()->backwardPass(1.0, grad);
			 };
		}

    protected:
        int nVarsForMatrices() const;

    private:
        std::vector<CompGraph::GNScalarVariable *> mScalarVariables;
        std::vector<CompGraph::GNMatrixVariable *> mMatrixVariables;
        std::vector<CompGraph::GraphNode *> mGraphNodes;
    };

}
#endif //NUMGRIND_GRAPHMANAGER_H
