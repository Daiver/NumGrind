#ifndef NUMGRINDTEST01_GRAPHNODEFUNCSCALARSUM_H
#define NUMGRINDTEST01_GRAPHNODEFUNCSCALARSUM_H

#include "CGScalarFunction.h"

namespace NumGrind {
    namespace CompGraph {
        class CGScalarSum : public CGScalarFunction {
        public:
            CGScalarSum(CGScalarOutput *argA, CGScalarOutput *argB);

            void forwardPass(const Eigen::VectorXf &vars) override;

            virtual void backwardPass(const float sensitivity, Eigen::VectorXf &grad) override;

        };
    }
}

#endif //NUMGRINDTEST01_GRAPHNODEFUNCSCALARSUM_H
