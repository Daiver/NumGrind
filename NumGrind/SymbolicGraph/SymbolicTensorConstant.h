#ifndef NUMGRIND_SYMBOLICTENSORCONSTANT_H
#define NUMGRIND_SYMBOLICTENSORCONSTANT_H

#include "CompGraph/CGMatrixConstant.h"
#include "SymbolicTensorNode.h"

namespace NumGrind {
    namespace SymbolicGraph {
        class SymbolicTensorConstant : public SymbolicTensorNode {
        public:
            SymbolicTensorConstant(SymbolicGraphManagerAbstract *manager, CompGraph::CGMatrixConstant *graphNode);

            void setValue(const Eigen::VectorXf value);

        protected:
            CompGraph::CGMatrixConstant *mNodeConstant;
        };
    }
}

#endif //NUMGRIND_SYMBOLICTENSORCONSTANT_H
