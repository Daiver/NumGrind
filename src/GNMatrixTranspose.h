#ifndef NUMGRINDTEST01_GNMATRIXTRANSPOSE_H
#define NUMGRINDTEST01_GNMATRIXTRANSPOSE_H

#include "GNTensorOutput.h"

class GNMatrixTranspose : public GNTensorOutput{
public:
    GNMatrixTranspose(GNTensorOutput *arg): arg(arg)
    {
        assert(false);//Not implemented yet
    }

    virtual void forwardPass(const Eigen::VectorXf &vars) override
    {
        arg->forwardPass(vars);
        this->mValue = arg->value().transpose();
    }

    virtual void backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) override {

    }

    virtual const Eigen::MatrixXf &value() const override {
        return mValue;
    }

    virtual std::string toString() const override
    {
        return "";
    }

private:
    GNTensorOutput *arg;
    Eigen::MatrixXf mValue;
};


#endif //NUMGRINDTEST01_GNMATRIXTRANSPOSE_H
