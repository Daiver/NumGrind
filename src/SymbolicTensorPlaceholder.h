#ifndef NUMGRIND_SYMBOLICTENSORPLACEHOLDER_H
#define NUMGRIND_SYMBOLICTENSORPLACEHOLDER_H

#include "SymbolicTensorNode.h"

#include "GNMatrixVariable.h"

class SymbolicTensorPlaceholder : public SymbolicTensorNode {
public:
    SymbolicTensorPlaceholder(GraphManagerAbstract *manager, GNMatrixVariable *graphNode, const bool isVariable = false);

    void setValue(const Eigen::MatrixXf &value) { this->mNodeVariable->setValue(value); }

    bool isVariable() const { return mIsVariable; }

protected:
    bool mIsVariable;
    GNMatrixVariable *mNodeVariable;
};


#endif //NUMGRIND_SYMBOLICTENSORPLACEHOLDER_H
