#ifndef NUMGRINDTEST01_GRAPHNODEVECTORELEMENTWISESUM_H
#define NUMGRINDTEST01_GRAPHNODEVECTORELEMENTWISESUM_H

#include "GNTensorOutput.h"

class GNVectorElementWiseSum : public GNTensorOutput
{
public:
    GNVectorElementWiseSum(GNTensorOutput *arg1, GNTensorOutput *arg2): arg1(arg1), arg2(arg2)
    {

    }

    virtual void forwardPass(const Eigen::VectorXf &vars) override
    {
        arg1->forwardPass(vars);
        arg2->forwardPass(vars);
        this->mValue = arg1->value().array() + arg2->value().array();
    }

    virtual void backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) override {
        this->arg1->backwardPass(sensitivity, grad);
        this->arg2->backwardPass(sensitivity, grad);
    }

    virtual const Eigen::MatrixXf &value() const override {
        return mValue;
    }

    virtual std::string toString() const override
    {
        return "";
    }

private:
    GNTensorOutput *arg1;
    GNTensorOutput *arg2;
    Eigen::MatrixXf mValue;
};


#endif //NUMGRINDTEST01_GRAPHNODEVECTORELEMENTWISESUM_H
