#include "SymbolicScalarPlaceholder.h"

SymbolicScalarPlaceholder::SymbolicScalarPlaceholder(GraphManagerAbstract *manager, GNScalarVariable *variable, const bool isVariable)
        : SymbolicScalarNode(manager, variable), mIsVariable(isVariable), mNodeVariable(variable)
{

}
