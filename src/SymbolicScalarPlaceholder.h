#ifndef NUMGRIND_SYMBOLICSCALARPLACEHOLDER_H
#define NUMGRIND_SYMBOLICSCALARPLACEHOLDER_H

#include "SymbolicScalarNode.h"
#include "CompGraph/GNScalarVariable.h"

namespace NumGrind {
    class SymbolicScalarPlaceholder : public SymbolicScalarNode {
    public:
        SymbolicScalarPlaceholder(GraphManagerAbstract *manager, CompGraph::GNScalarVariable *variable, const bool isVariable);

        bool isVariable() const { return mIsVariable; }

        void setValue(const float value) { this->mNodeVariable->setValue(value); }

    protected:
        bool mIsVariable;
        CompGraph::GNScalarVariable *mNodeVariable;
    };
}

#endif //NUMGRIND_SYMBOLICSCALARPLACEHOLDER_H
