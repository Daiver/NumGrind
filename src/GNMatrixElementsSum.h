#ifndef NUMGRINDTEST01_GNMATRIXELEMENTSSUM_H
#define NUMGRINDTEST01_GNMATRIXELEMENTSSUM_H

#include "GNTensorOutput.h"
#include "GNScalarOutput.h"

class GNMatrixElementsSum : public GNScalarOutput{
public:
    GNMatrixElementsSum(GNTensorOutput *arg): arg(arg)
    {

    }

    virtual void forwardPass(const Eigen::VectorXf &vars) override
    {
        arg->forwardPass(vars);
        auto res = arg->value();

        this->mValue = 0;
        for(int i = 0; i < res.rows(); ++i)
            for(int j = 0; j < res.cols(); ++j)
                this->mValue += res(i, j);
    }

    virtual void backwardPass(const float sensitivity, Eigen::VectorXf &grad) override {
        const auto res = arg->value();
        const int nRows = res.rows();
        const int nCols = res.cols();
        const auto sens = Eigen::MatrixXf::Constant(nRows, nCols, sensitivity);
        this->arg->backwardPass(sens, grad);
    }

    virtual float value() const override {
        return mValue;
    }

    virtual std::string toString() const override
    {
        return "";
    }

private:
    GNTensorOutput *arg;
    float mValue;
};

#endif //NUMGRINDTEST01_GNMATRIXELEMENTSSUM_H
