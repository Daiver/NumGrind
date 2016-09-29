//
// Created by daiver on 25.09.16.
//

#include "SymbolicTensorVariable.h"

using namespace NumGrind::SymbolicGraph;
using namespace NumGrind::CompGraph;

SymbolicTensorVariable::SymbolicTensorVariable(SymbolicGraphManagerAbstract *manager, CGMatrixVariable *graphNode, const bool isVariable)
        : SymbolicTensorNode(manager, graphNode), mIsVariable(isVariable)
{

}
