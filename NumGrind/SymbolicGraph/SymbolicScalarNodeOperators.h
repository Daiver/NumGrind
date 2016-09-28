#ifndef NUMGRIND_SYMBOLICSCALARNODEOPERATORS_H
#define NUMGRIND_SYMBOLICSCALARNODEOPERATORS_H

#include "SymbolicScalarNode.h"
#include "SymbolicScalarPlaceholder.h"

#include "CompGraph/CGScalarVariable.h"
#include "CompGraph/CGScalarConst.h"
#include "CompGraph/CGScalarSum.h"
#include "CompGraph/CGScalarMult.h"
#include "CompGraph/CGScalarSub.h"
#include "CompGraph/CGScalarConst.h"

namespace NumGrind {
    namespace SymbolicGraph {

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

