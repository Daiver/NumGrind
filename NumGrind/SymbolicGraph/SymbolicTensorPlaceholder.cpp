//
// Created by daiver on 25.09.16.
//

#include "SymbolicTensorPlaceholder.h"

using namespace NumGrind::SymbolicGraph;
using namespace NumGrind::CompGraph;

SymbolicTensorPlaceholder::SymbolicTensorPlaceholder(SymbolicGraphManagerAbstract *manager, CGMatrixVariable *graphNode, const bool isVariable)
        : SymbolicTensorNode(manager, graphNode), mIsVariable(isVariable)
{

}
