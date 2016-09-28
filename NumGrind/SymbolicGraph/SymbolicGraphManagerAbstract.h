#ifndef NUMGRIND_GRAPHMANAGERABSTRACT_H
#define NUMGRIND_GRAPHMANAGERABSTRACT_H

#include "CompGraph/CompGraphNode.h"

namespace NumGrind {
    namespace SymbolicGraph {
        class SymbolicGraphManagerAbstract {
        public:
            virtual void addGraphNode(CompGraph::CompGraphNode *node) = 0;

        };
    }
}

#endif //NUMGRIND_GRAPHMANAGERABSTRACT_H
