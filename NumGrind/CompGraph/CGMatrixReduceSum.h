#ifndef NUMGRINDTEST01_GNMATRIXELEMENTSSUM_H
#define NUMGRINDTEST01_GNMATRIXELEMENTSSUM_H

#include "CGTensorOutput.h"
#include "CGScalarOutput.h"

namespace NumGrind {
    namespace CompGraph {

        class CGMatrixReduceSum : public CGScalarOutput {
        public:
            CGMatrixReduceSum(CGTensorOutput *arg);

            virtual void forwardPass(const Eigen::VectorXf &vars) override;

            virtual void backwardPass(const float sensitivity, Eigen::VectorXf &grad) override;

            virtual float value() const override {
                return mValue;
            }

        private:
            CGTensorOutput *arg;
            float mValue;
        };

    }
}
#endif //NUMGRINDTEST01_GNMATRIXELEMENTSSUM_H
