#ifndef NUMGRINDTEST01_GNMATRIXTRANSPOSE_H
#define NUMGRINDTEST01_GNMATRIXTRANSPOSE_H

#include "CGTensorOutput.h"

namespace NumGrind {
    namespace CompGraph {
        class CGMatrixTranspose : public CGTensorOutput {
        public:
            CGMatrixTranspose(CGTensorOutput *arg) : arg(arg) {
                assert(false);//Not implemented yet
            }

            virtual void forwardPass(const Eigen::VectorXf &vars) override {
                arg->forwardPass(vars);
                this->mValue = arg->value().transpose();
            }

            virtual void backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) override {

            }

            virtual const Eigen::MatrixXf &value() const override {
                return mValue;
            }

        private:
            CGTensorOutput *arg;
            Eigen::MatrixXf mValue;
        };

    }
}

#endif //NUMGRINDTEST01_GNMATRIXTRANSPOSE_H
