#ifndef NUMGRINDTEST01_GRAPHNODEFUNCSCALARSUM_H
#define NUMGRINDTEST01_GRAPHNODEFUNCSCALARSUM_H

#include "GNScalarFunction.h"

namespace NumGrind {
    namespace CompGraph {
        class GNScalarSum : public GNScalarFunction {
        public:
            GNScalarSum(GNScalarOutput *argA, GNScalarOutput *argB) {
                this->arguments.push_back(argA);
                this->arguments.push_back(argB);
            }

            void forwardPass(const Eigen::VectorXf &vars) {
                float res = 0.0;
                for (GNScalarOutput *arg : this->arguments) {
                    arg->forwardPass(vars);
                    res += arg->value();
                }
                this->mValue = res;
            }

            virtual void backwardPass(const float sensitivity, Eigen::VectorXf &grad) override {
                for (GNScalarOutput *arg : arguments) {
                    arg->backwardPass(sensitivity, grad);
                }
            }

        };
    }
}

#endif //NUMGRINDTEST01_GRAPHNODEFUNCSCALARSUM_H
