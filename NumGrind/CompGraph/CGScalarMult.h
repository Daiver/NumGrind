#ifndef NUMGRINDTEST01_GRAPHNODEFUNCSCALARMULT_H
#define NUMGRINDTEST01_GRAPHNODEFUNCSCALARMULT_H

#include "CGScalarFunction.h"

namespace NumGrind {
    namespace CompGraph {
        class CGScalarMult : public CGScalarFunction {
        public:
            CGScalarMult(CGScalarOutput *argA, CGScalarOutput *argB);

            void forwardPass(const Eigen::VectorXf &vars) override;

            virtual void backwardPass(const float sensitivity, Eigen::VectorXf &grad) override;

        };

    }
}

#endif //NUMGRINDTEST01_GRAPHNODEFUNCSCALARMULT_H
