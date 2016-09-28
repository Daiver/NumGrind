#ifndef NUMGRIND_GNSCALARSUB_H
#define NUMGRIND_GNSCALARSUB_H

#include "CGScalarOutput.h"

namespace NumGrind {
    namespace CompGraph {
        class CGScalarSub : public CGScalarOutput {
        public:
            CGScalarSub(CGScalarOutput *arg1, CGScalarOutput *arg2) : arg1(arg1), arg2(arg2) {

            }

            virtual void forwardPass(const Eigen::VectorXf &vars) override {
                arg1->forwardPass(vars);
                arg2->forwardPass(vars);
                this->mValue = arg1->value() - arg2->value();
            }

            virtual void backwardPass(const float sensitivity, Eigen::VectorXf &grad) override {
                arg1->backwardPass(sensitivity, grad);
                arg2->backwardPass(-sensitivity, grad);
            }

            virtual float value() const override {
                return mValue;
            }

        private:
            float mValue;
            CGScalarOutput *arg1;
            CGScalarOutput *arg2;
        };

    }
}

#endif //NUMGRIND_GNSCALARSUB_H
