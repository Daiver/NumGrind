#ifndef NUMGRIND_GNMATRIXSUB_H
#define NUMGRIND_GNMATRIXSUB_H

#include "CGTensorOutput.h"

namespace NumGrind {
    namespace CompGraph {

        class CGMatrixSub : public CGTensorOutput {
        public:
            CGMatrixSub(CGTensorOutput *arg1, CGTensorOutput *arg2);

            virtual void forwardPass(const Eigen::VectorXf &vars) override;

            virtual void backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) override;

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

