#ifndef NUMGRIND_SYMBOLICGRAPHNODE_H
#define NUMGRIND_SYMBOLICGRAPHNODE_H

#include "SymbolicGraphManagerAbstract.h"

namespace NumGrind {
    namespace SymbolicGraph {
        class SymbolicGraphNode {
        public:
            SymbolicGraphNode(SymbolicGraphManagerAbstract *manager);

            virtual ~SymbolicGraphNode() = 0;

            SymbolicGraphManagerAbstract *manager();

        protected:
            SymbolicGraphManagerAbstract *mManager;
        };
    }
}

#endif //NUMGRIND_SYMBOLICGRAPHNODE_H
