#ifndef NUMGRIND_SYMBOLICSCALARNODEOPERATORS_H
#define NUMGRIND_SYMBOLICSCALARNODEOPERATORS_H

#include "SymbolicScalarNode.h"
#include "SymbolicScalarPlaceholder.h"

#include "GNScalarVariable.h"
#include "GNScalarConst.h"
#include "GNScalarSum.h"
#include "GNScalarMult.h"
#include "GNScalarSub.h"
#include "GNScalarConst.h"

namespace NumGrind {
    namespace SymbolicNodeOps {

        SymbolicScalarNode operator+(SymbolicScalarNode a, SymbolicScalarNode b);

        SymbolicScalarNode operator-(SymbolicScalarNode a, SymbolicScalarNode b);

        SymbolicScalarNode operator*(SymbolicScalarNode a, SymbolicScalarNode b);

        SymbolicScalarNode operator+(const float a, SymbolicScalarNode b);

        SymbolicScalarNode operator-(const float a, SymbolicScalarNode b);

        SymbolicScalarNode operator*(const float a, SymbolicScalarNode b);

        SymbolicScalarNode operator+(SymbolicScalarNode a, const float b);

        SymbolicScalarNode operator-(SymbolicScalarNode a, const float b);

        SymbolicScalarNode operator*(SymbolicScalarNode a, const float b);

        SymbolicScalarNode operator/(SymbolicScalarNode a, const float b);
    }
}

#endif //NUMGRIND_SCALARNODEOPERATORS_H

