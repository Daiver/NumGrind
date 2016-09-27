//
// Created by daiver on 25.09.16.
//

#include "SymbolicTensorPlaceholder.h"

using namespace NumGrind;
using namespace NumGrind::CompGraph;

SymbolicTensorPlaceholder::SymbolicTensorPlaceholder(GraphManagerAbstract *manager, GNMatrixVariable *graphNode, const bool isVariable)
        : SymbolicTensorNode(manager, graphNode), mIsVariable(isVariable)
{

}
