#ifndef NUMGRINDTEST01_GNVECTORCONSTANT_H
#define NUMGRINDTEST01_GNVECTORCONSTANT_H

#include "GNTensorOutput.h"

namespace NumGrind {
    namespace CompGraph {
        class GNMatrixConstant : public GNTensorOutput {
        public:
            GNMatrixConstant(const Eigen::MatrixXf &value) : mValue(value) {

            }

            virtual void forwardPass(const Eigen::VectorXf &vars) override {}

            virtual void backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) override {}

            virtual const Eigen::MatrixXf &value() const override {
                return mValue;
            }

        private:
            Eigen::MatrixXf mValue;
        };
    }
}

#endif //NUMGRINDTEST01_GNVECTORCONSTANT_H
