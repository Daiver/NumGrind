#ifndef NUMGRINDTEST01_GNMATRIXMAPFUNCTION_H
#define NUMGRINDTEST01_GNMATRIXMAPFUNCTION_H

#include "CGTensorOutput.h"
namespace NumGrind {
    namespace CompGraph {
        template<typename Scalar, Scalar Func(Scalar), Scalar Der(Scalar)>
        class CGMatrixMapUnaryFunction : public CGTensorOutput {
        public:
            CGMatrixMapUnaryFunction(CGTensorOutput *arg) : arg(arg) {}

            virtual void forwardPass(const Eigen::VectorXf &vars) override {
                this->arg->forwardPass(vars);
                auto res = arg->value();
                this->mValue.resize(res.rows(), res.cols());
                for (int i = 0; i < mValue.rows(); ++i)
                    for (int j = 0; j < mValue.cols(); ++j)
                        mValue(i, j) = Func(res(i, j));
            }

            virtual void backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) override {
                auto res = arg->value();
                Eigen::MatrixXf der(res.rows(), res.cols());
                for (int i = 0; i < mValue.rows(); ++i)
                    for (int j = 0; j < mValue.cols(); ++j)
                        der(i, j) = Der(res(i, j));
                arg->backwardPass(sensitivity.array() * der.array(), grad);
            }

            virtual const Eigen::MatrixXf &value() const override {
                return mValue;
            }

        private:
            Eigen::MatrixXf mValue;
            CGTensorOutput *arg;
        };

    }
}
#endif //NUMGRINDTEST01_GNMATRIXMAPFUNCTION_H
