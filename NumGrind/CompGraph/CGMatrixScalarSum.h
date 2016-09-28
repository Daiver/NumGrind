#ifndef NUMGRIND_GNVECTORSCALARSUM_H
#define NUMGRIND_GNVECTORSCALARSUM_H

#include "CGScalarOutput.h"
#include "CGTensorOutput.h"

namespace NumGrind {
    namespace CompGraph {

        class CGMatrixScalarSum : public CGTensorOutput {
        public:
            CGMatrixScalarSum(CGTensorOutput *arg1, CGScalarOutput *arg2) : arg1(arg1), arg2(arg2) {

            }

            virtual void forwardPass(const Eigen::VectorXf &vars) override {
                arg1->forwardPass(vars);
                arg2->forwardPass(vars);
                mValue = arg1->value().array() + arg2->value();
            }

            virtual void backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) override {
                arg1->backwardPass(sensitivity, grad);
                arg2->backwardPass(sensitivity.array().sum(), grad);
            }

            virtual const Eigen::MatrixXf &value() const override {
                return mValue;
            }

        private:
            Eigen::MatrixXf mValue;
            CGTensorOutput *arg1;
            CGScalarOutput *arg2;
        };

    }
}
#endif //NUMGRIND_GNVECTORSCALARSUM_H
