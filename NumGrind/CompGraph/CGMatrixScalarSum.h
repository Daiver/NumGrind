#ifndef NUMGRIND_GNVECTORSCALARSUM_H
#define NUMGRIND_GNVECTORSCALARSUM_H

#include "CGScalarOutput.h"
#include "CGTensorOutput.h"

namespace NumGrind {
    namespace CompGraph {

        class CGMatrixScalarSum : public CGTensorOutput {
        public:
            CGMatrixScalarSum(CGTensorOutput *arg1, CGScalarOutput *arg2);

            virtual void forwardPass(const Eigen::VectorXf &vars) override;

            virtual void backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) override;

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
