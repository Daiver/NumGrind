#ifndef NUMGRINDTEST01_GNMATRIXPRODUCT_H
#define NUMGRINDTEST01_GNMATRIXPRODUCT_H

#include "GNTensorOutput.h"

namespace NumGrind {
    namespace CompGraph {
        class GNMatrixProduct : public GNTensorOutput {
        public:
            GNMatrixProduct(GNTensorOutput *arg1, GNTensorOutput *arg2) : arg1(arg1), arg2(arg2) {
//        assert(false);//Not implemented yet
            }

            void forwardPass(const Eigen::VectorXf &vars) {
                arg1->forwardPass(vars);
                arg2->forwardPass(vars);
                auto res1 = arg1->value();
                auto res2 = arg2->value();
                assert(res1.cols() == res2.rows());
                this->mValue = res1 * res2;
            }

            virtual void backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) override {
                auto res1 = arg1->value();
                auto res2 = arg2->value();

//        int res1R = res1.rows();
//        int res1C = res1.cols();
//
//        int res2R = res2.rows();
//        int res2C = res2.cols();
//
//        int sensR = sensitivity.rows();
//        int sensC = sensitivity.cols();

                arg1->backwardPass((sensitivity * res2.transpose()), grad);
                arg2->backwardPass((res1.transpose() * sensitivity), grad);
            }

            const Eigen::MatrixXf &value() const { return this->mValue; }

        private:
            Eigen::MatrixXf mValue;
            GNTensorOutput *arg1;
            GNTensorOutput *arg2;
        };
    }
}

#endif //NUMGRINDTEST01_GNMATRIXPRODUCT_H

