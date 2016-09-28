#ifndef NUMGRINDTEST01_GRAPHNODEDOTPRODUCT_H
#define NUMGRINDTEST01_GRAPHNODEDOTPRODUCT_H

#include "CGScalarOutput.h"
#include "CGTensorOutput.h"

namespace NumGrind {
    namespace CompGraph {

        class CGDotProduct : public CGScalarOutput {
        public:
            CGDotProduct(CGTensorOutput *arg1, CGTensorOutput *arg2) : arg1(arg1), arg2(arg2) {

            }

            virtual void forwardPass(const Eigen::VectorXf &vars) override {
                arg1->forwardPass(vars);
                arg2->forwardPass(vars);

                auto res = arg1->value().transpose() * arg2->value();
                assert(res.rows() == 1);
                this->mValue = res(0, 0);
            }

            virtual void backwardPass(const float sensitivity, Eigen::VectorXf &grad) override {
                auto res1 = arg1->value();
                auto res2 = arg2->value();
                this->arg1->backwardPass(sensitivity * res2, grad);
                this->arg2->backwardPass(sensitivity * res1, grad);
            }

            virtual float value() const override {
                return mValue;
            }

        private:
            CGTensorOutput *arg1;
            CGTensorOutput *arg2;
            float mValue;
        };


    }
}

#endif //NUMGRINDTEST01_GRAPHNODEDOTPRODUCT_H
