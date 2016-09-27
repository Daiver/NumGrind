#ifndef NUMGRIND_SYMBOLICSCALARNODEOPERATORS_H
#define NUMGRIND_SYMBOLICSCALARNODEOPERATORS_H

#include "SymbolicScalarNode.h"
#include "SymbolicScalarPlaceholder.h"

#include "CompGraph/GNScalarVariable.h"
#include "CompGraph/GNScalarConst.h"
#include "CompGraph/GNScalarSum.h"
#include "CompGraph/GNScalarMult.h"
#include "CompGraph/GNScalarSub.h"
#include "CompGraph/GNScalarConst.h"

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

