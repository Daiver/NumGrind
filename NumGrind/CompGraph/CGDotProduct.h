#ifndef NUMGRINDTEST01_GRAPHNODEDOTPRODUCT_H
#define NUMGRINDTEST01_GRAPHNODEDOTPRODUCT_H

#include "CGScalarOutput.h"
#include "CGTensorOutput.h"

namespace NumGrind {
    namespace CompGraph {

        class CGDotProduct : public CGScalarOutput {
        public:
            CGDotProduct(CGTensorOutput *arg1, CGTensorOutput *arg2);

            virtual void forwardPass(const Eigen::VectorXf &vars) override;

            virtual void backwardPass(const float sensitivity, Eigen::VectorXf &grad) override;

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
