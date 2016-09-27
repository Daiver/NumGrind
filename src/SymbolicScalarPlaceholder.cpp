#include "SymbolicScalarPlaceholder.h"

NumGrind::SymbolicScalarPlaceholder::SymbolicScalarPlaceholder(GraphManagerAbstract *manager, CompGraph::GNScalarVariable *variable, const bool isVariable)
        : SymbolicScalarNode(manager, variable), mIsVariable(isVariable), mNodeVariable(variable)
{

}
