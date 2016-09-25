#ifndef NUMGRIND_SYMBOLICSCALARNODEOPERATORS_H
#define NUMGRIND_SYMBOLICSCALARNODEOPERATORS_H

#include "SymbolicScalarNode.h"
#include "SymbolicScalarPlaceholder.h"

#include "GNScalarVariable.h"
#include "GNScalarConst.h"
#include "GNScalarSum.h"
#include "GNScalarMult.h"
#include "GNScalarSub.h"

namespace SymbolicScalarNodeOperators {

    SymbolicScalarNode operator+(SymbolicScalarNode &a, SymbolicScalarNode &b);
    SymbolicScalarNode operator-(SymbolicScalarNode &a, SymbolicScalarNode &b);
    SymbolicScalarNode operator*(SymbolicScalarNode &a, SymbolicScalarNode &b);
};


#endif //NUMGRIND_SCALARNODEOPERATORS_H
