#include "SymbolicScalarPlaceholder.h"

using namespace NumGrind;
using namespace NumGrind::SymbolicGraph;

NumGrind::SymbolicGraph::SymbolicScalarPlaceholder::SymbolicScalarPlaceholder(GraphManagerAbstract *manager, CompGraph::GNScalarVariable *variable, const bool isVariable)
        : SymbolicScalarNode(manager, variable), mIsVariable(isVariable), mNodeVariable(variable)
{

}
