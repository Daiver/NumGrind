#ifndef NUMGRIND_GNSCALARSUB_H
#define NUMGRIND_GNSCALARSUB_H

#include "GNScalarOutput.h"

namespace NumGrind {
    namespace CompGraph {


        class GNScalarSub : public GNScalarOutput {
        public:
            GNScalarSub(GNScalarOutput *arg1, GNScalarOutput *arg2) : arg1(arg1), arg2(arg2) {

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

            virtual std::string toString() const override {
                return "";
            }

        private:
            float mValue;
            GNScalarOutput *arg1;
            GNScalarOutput *arg2;
        };

    }
}

#endif //NUMGRIND_GNSCALARSUB_H
