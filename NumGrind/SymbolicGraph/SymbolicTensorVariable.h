#ifndef NUMGRIND_SYMBOLICTENSORPLACEHOLDER_H
#define NUMGRIND_SYMBOLICTENSORPLACEHOLDER_H

#include "SymbolicTensorNode.h"

#include "CompGraph/CGMatrixVariable.h"

namespace NumGrind {
    namespace SymbolicGraph {
        class SymbolicTensorVariable : public SymbolicTensorNode {
        public:
            SymbolicTensorVariable(SymbolicGraphManagerAbstract *manager, CompGraph::CGMatrixVariable *graphNode,
                                      const bool isVariable = false);

            void setValue(const Eigen::MatrixXf &value) { this->mNodeVariable->setValue(value); }

            bool isVariable() const { return mIsVariable; }

        protected:
            bool mIsVariable;
            CompGraph::CGMatrixVariable *mNodeVariable;
        };
    }
}

#endif //NUMGRIND_SYMBOLICTENSORPLACEHOLDER_H
