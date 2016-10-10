#ifndef NUMGRINDTEST01_GRAPHNODEVECTORELEMENTWISEPRODUCT_H
#define NUMGRINDTEST01_GRAPHNODEVECTORELEMENTWISEPRODUCT_H

#include "CGTensorOutput.h"

namespace NumGrind {
    namespace CompGraph {

        class CGMatrixElementWiseProduct : public CGTensorOutput {
        public:
            CGMatrixElementWiseProduct(CGTensorOutput *arg1, CGTensorOutput *arg2);

            virtual void forwardPass(const Eigen::VectorXf &vars) override;

            virtual void backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) override;

            virtual const Eigen::MatrixXf &value() const override {
                return mValue;
            }

        private:
            CGTensorOutput *arg1;
            CGTensorOutput *arg2;
            Eigen::MatrixXf mValue;
        };
    }
}

#endif //NUMGRINDTEST01_GRAPHNODEVECTORELEMENTWISEPRODUCT_H
