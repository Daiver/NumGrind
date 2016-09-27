#ifndef NUMGRIND_SYMBOLICTENSORPLACEHOLDER_H
#define NUMGRIND_SYMBOLICTENSORPLACEHOLDER_H

#include "SymbolicTensorNode.h"

#include "CompGraph/GNMatrixVariable.h"

namespace NumGrind {
    class SymbolicTensorPlaceholder : public SymbolicTensorNode {
    public:
        SymbolicTensorPlaceholder(GraphManagerAbstract *manager, CompGraph::GNMatrixVariable *graphNode,
                                  const bool isVariable = false);

        void setValue(const Eigen::MatrixXf &value) { this->mNodeVariable->setValue(value); }

        bool isVariable() const { return mIsVariable; }

    protected:
        bool mIsVariable;
        CompGraph::GNMatrixVariable *mNodeVariable;
    };
}

#endif //NUMGRIND_SYMBOLICTENSORPLACEHOLDER_H
