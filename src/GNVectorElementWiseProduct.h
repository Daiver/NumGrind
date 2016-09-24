#ifndef NUMGRINDTEST01_GRAPHNODEVECTORELEMENTWISEPRODUCT_H
#define NUMGRINDTEST01_GRAPHNODEVECTORELEMENTWISEPRODUCT_H

#include "GNTensorOutput.h"

class GNVectorElementWiseProduct : public GNTensorOutput
{
public:
    GNVectorElementWiseProduct(GNTensorOutput *arg1, GNTensorOutput *arg2): arg1(arg1), arg2(arg2)
    {

    }

    virtual void forwardPass(const Eigen::VectorXf &vars) override
    {
        arg1->forwardPass(vars);
        arg2->forwardPass(vars);
        this->mValue = arg1->value().array() * arg2->value().array();
    }

    virtual void backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) override {
        auto res1 = arg1->value();
        auto res2 = arg2->value();
        this->arg1->backwardPass(sensitivity.array() * res2.array(), grad);
        this->arg2->backwardPass(sensitivity.array() * res1.array(), grad);
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


#endif //NUMGRINDTEST01_GRAPHNODEVECTORELEMENTWISEPRODUCT_H
