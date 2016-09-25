//
// Created by daiver on 25.09.16.
//

#include "SymbolicTensorPlaceholder.h"

SymbolicTensorPlaceholder::SymbolicTensorPlaceholder(GraphManagerAbstract *manager, GNMatrixVariable *graphNode, const bool isVariable)
        : SymbolicTensorNode(manager, graphNode), mIsVariable(isVariable)
{

}
