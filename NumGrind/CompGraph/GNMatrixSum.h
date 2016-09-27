#ifndef NUMGRINDTEST01_GRAPHNODEVECTORELEMENTWISESUM_H
#define NUMGRINDTEST01_GRAPHNODEVECTORELEMENTWISESUM_H

#include "GNTensorOutput.h"

namespace NumGrind {
    namespace CompGraph {
        class GNMatrixSum : public GNTensorOutput {
        public:
            GNMatrixSum(GNTensorOutput *arg1, GNTensorOutput *arg2) : arg1(arg1), arg2(arg2) {

            }

            virtual void forwardPass(const Eigen::VectorXf &vars) override {
                arg1->forwardPass(vars);
                arg2->forwardPass(vars);
                auto res1 = arg1->value();
                auto res2 = arg2->value();
                if (res1.rows() == res2.rows() && res1.cols() == res2.cols()) {
                    this->mValue = res1.array() + res2.array();
                } else if (res1.cols() == res2.cols() && res2.rows() == 1) {
                    mValue = res1;
                    for (int i = 0; i < res1.rows(); ++i)
                        mValue.row(i) += res2;
                }
            }

            virtual void backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) override {
                auto res1 = arg1->value();
                auto res2 = arg2->value();
                if (res1.rows() == res2.rows() && res1.cols() == res2.cols()) {
                    this->arg1->backwardPass(sensitivity, grad);
                    this->arg2->backwardPass(sensitivity, grad);
                } else if (res1.cols() == res2.cols() && res2.rows() == 1) {
                    this->arg1->backwardPass(sensitivity, grad);
                    this->arg2->backwardPass(sensitivity.colwise().sum(), grad);
                }
            }

            virtual const Eigen::MatrixXf &value() const override {
                return mValue;
            }

        private:
            GNTensorOutput *arg1;
            GNTensorOutput *arg2;
            Eigen::MatrixXf mValue;
        };
    }
}

#endif //NUMGRINDTEST01_GRAPHNODEVECTORELEMENTWISESUM_H
