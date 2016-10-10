#ifndef NUMGRINDTEST01_GRAPHNODEFUNCSCALARMULT_H
#define NUMGRINDTEST01_GRAPHNODEFUNCSCALARMULT_H

#include "CGScalarFunction.h"

namespace NumGrind {
    namespace CompGraph {
        class CGScalarMult : public CGScalarFunction {
        public:
            CGScalarMult(CGScalarOutput *argA, CGScalarOutput *argB) {
                this->arguments.push_back(argA);
                this->arguments.push_back(argB);
            }

            void forwardPass(const Eigen::VectorXf &vars) override {
                float res = 1.0;
                for (CGScalarOutput *arg : this->arguments) {
                    arg->forwardPass(vars);
                    res *= arg->value();
                }
                this->mValue = res;
            }

            virtual void backwardPass(const float sensitivity, Eigen::VectorXf &grad) override {
                assert(arguments.size() == 2);
                arguments[0]->backwardPass(sensitivity * arguments[1]->value(), grad);
                arguments[1]->backwardPass(sensitivity * arguments[0]->value(), grad);
            }

        };

    }
}

#endif //NUMGRINDTEST01_GRAPHNODEFUNCSCALARMULT_H
