#ifndef NUMGRIND_GNMATRIXSUB_H
#define NUMGRIND_GNMATRIXSUB_H

#include "CGTensorOutput.h"

namespace NumGrind {
    namespace CompGraph {

        class CGMatrixSub : public CGTensorOutput {
        public:
            CGMatrixSub(CGTensorOutput *arg1, CGTensorOutput *arg2) : arg1(arg1), arg2(arg2) {

            }

            virtual void forwardPass(const Eigen::VectorXf &vars) override {
                arg1->forwardPass(vars);
                arg2->forwardPass(vars);
                auto res1 = arg1->value();
                auto res2 = arg2->value();
                mValue = res1 - res2;
            }

            virtual void backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) override {
                arg1->backwardPass(sensitivity, grad);
                arg2->backwardPass(-sensitivity, grad);
            }

            virtual const Eigen::MatrixXf &value() const override {
                return mValue;
            }

        private:
            Eigen::MatrixXf mValue;

            CGTensorOutput *arg1;
            CGTensorOutput *arg2;
        };

    }
}
#endif //NUMGRIND_GNMATRIXSUB_H
