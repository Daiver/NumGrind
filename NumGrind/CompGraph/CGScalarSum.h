#ifndef NUMGRINDTEST01_GRAPHNODEFUNCSCALARSUM_H
#define NUMGRINDTEST01_GRAPHNODEFUNCSCALARSUM_H

#include "CGScalarFunction.h"

namespace NumGrind {
    namespace CompGraph {
        class CGScalarSum : public CGScalarFunction {
        public:
            CGScalarSum(CGScalarOutput *argA, CGScalarOutput *argB) {
                this->arguments.push_back(argA);
                this->arguments.push_back(argB);
            }

            void forwardPass(const Eigen::VectorXf &vars) override {
                float res = 0.0;
                for (CGScalarOutput *arg : this->arguments) {
                    arg->forwardPass(vars);
                    res += arg->value();
                }
                this->mValue = res;
            }

            virtual void backwardPass(const float sensitivity, Eigen::VectorXf &grad) override {
                for (CGScalarOutput *arg : arguments) {
                    arg->backwardPass(sensitivity, grad);
                }
            }

        };
    }
}

#endif //NUMGRINDTEST01_GRAPHNODEFUNCSCALARSUM_H
