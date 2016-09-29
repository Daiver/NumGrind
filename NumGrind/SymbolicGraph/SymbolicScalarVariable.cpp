#include "SymbolicScalarVariable.h"

using namespace NumGrind;
using namespace NumGrind::SymbolicGraph;

NumGrind::SymbolicGraph::SymbolicScalarVariable::SymbolicScalarVariable(SymbolicGraphManagerAbstract *manager, CompGraph::CGScalarVariable *variable, const bool isVariable)
        : SymbolicScalarNode(manager, variable), mIsVariable(isVariable), mNodeVariable(variable)
{

}
