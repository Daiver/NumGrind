#ifndef NUMGRIND_SYMBOLICSCALARPLACEHOLDER_H
#define NUMGRIND_SYMBOLICSCALARPLACEHOLDER_H

#include "SymbolicScalarNode.h"
#include "CompGraph/CGScalarVariable.h"

namespace NumGrind {
    namespace SymbolicGraph {
        class SymbolicScalarPlaceholder : public SymbolicScalarNode {
        public:
            SymbolicScalarPlaceholder(SymbolicGraphManagerAbstract *manager, CompGraph::CGScalarVariable *variable,
                                      const bool isVariable);

            bool isVariable() const { return mIsVariable; }

            void setValue(const float value) { this->mNodeVariable->setValue(value); }

        protected:
            bool mIsVariable;
            CompGraph::CGScalarVariable *mNodeVariable;
        };
    }
}

#endif //NUMGRIND_SYMBOLICSCALARPLACEHOLDER_H
