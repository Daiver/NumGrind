#ifndef NUMGRINDTEST01_GNMATRIXPRODUCT_H
#define NUMGRINDTEST01_GNMATRIXPRODUCT_H

#include "CGTensorOutput.h"

namespace NumGrind {
    namespace CompGraph {
        class CGMatrixProduct : public CGTensorOutput {
        public:
            CGMatrixProduct(CGTensorOutput *arg1, CGTensorOutput *arg2);

            void forwardPass(const Eigen::VectorXf &vars) override;

            virtual void backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) override;

            const Eigen::MatrixXf &value() const override { return this->mValue; }

        private:
            Eigen::MatrixXf mValue;
            CGTensorOutput *arg1;
            CGTensorOutput *arg2;
        };
    }
}

#endif //NUMGRINDTEST01_GNMATRIXPRODUCT_H

