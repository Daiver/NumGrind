#ifndef NUMGRIND_SYMBOLICSCALARPLACEHOLDER_H
#define NUMGRIND_SYMBOLICSCALARPLACEHOLDER_H

#include "SymbolicScalarNode.h"
#include "GNScalarVariable.h"

class SymbolicScalarPlaceholder : public SymbolicScalarNode {
public:
    SymbolicScalarPlaceholder(GraphManagerAbstract *manager, GNScalarVariable *variable,
                                  const bool isVariable);

    bool isVariable() const { return mIsVariable; }

    void setValue(const float value) { this->mNodeVariable->setValue(value); }

protected:
    bool mIsVariable;
    GNScalarVariable *mNodeVariable;
};


#endif //NUMGRIND_SYMBOLICSCALARPLACEHOLDER_H
