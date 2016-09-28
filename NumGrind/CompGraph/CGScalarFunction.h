#ifndef NUMGRINDTEST01_GRAPHNODEFUNCTION_H
#define NUMGRINDTEST01_GRAPHNODEFUNCTION_H

#include <assert.h>

#include "CGScalarOutput.h"

namespace NumGrind {
    namespace CompGraph {
        class CGScalarFunction : public CGScalarOutput {
        public:
            float value() const { return this->mValue; }

        protected:
            std::vector<CGScalarOutput *> arguments;
            float mValue;
        };

    }
}

#endif //NUMGRINDTEST01_GRAPHNODEFUNCTION_H
