#ifndef NUMGRIND_GRAPHMANAGER_H
#define NUMGRIND_GRAPHMANAGER_H

#include <vector>
#include <functional>
#include "Eigen/Core"
#include "SymbolicGraph/SymbolicGraphManagerAbstract.h"
#include "SymbolicGraph/SymbolicScalarVariable.h"
#include "SymbolicGraph/SymbolicTensorVariable.h"
#include "SymbolicGraph/SymbolicTensorConstant.h"

namespace NumGrind {
    class GraphManager : public SymbolicGraph::SymbolicGraphManagerAbstract {
    public:
        GraphManager();

        ~GraphManager();

        virtual void addGraphNode(CompGraph::CompGraphNode *node) override;

        Eigen::VectorXf initializeVariables();

        static Eigen::VectorXf initializeGradient(const Eigen::VectorXf &vars);

        SymbolicGraph::SymbolicScalarVariable variable(const float val = 0.0);

        SymbolicGraph::SymbolicTensorVariable variable(const int nRows, const int nCols, const float val = 0.0);

        SymbolicGraph::SymbolicTensorVariable variable(const Eigen::MatrixXf &value);

        SymbolicGraph::SymbolicTensorConstant constant(const Eigen::MatrixXf &value);

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
                     grad.fill(0);
					 func->node()->forwardPass(vars);
					 func->node()->backwardPass(1.0, grad);
			 };
		}

    protected:
        int nVarsForMatrices() const;

    private:
        std::vector<CompGraph::CGScalarVariable *> mScalarVariables;
        std::vector<CompGraph::CGMatrixVariable *> mMatrixVariables;
        std::vector<CompGraph::CompGraphNode *> mGraphNodes;
    };

}
#endif //NUMGRIND_GRAPHMANAGER_H
