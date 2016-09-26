#ifndef NUMGRIND_GNMATRIXSUB_H
#define NUMGRIND_GNMATRIXSUB_H

#include "GNTensorOutput.h"

class GNMatrixSub : public GNTensorOutput{
public:
    GNMatrixSub(GNTensorOutput *arg1, GNTensorOutput *arg2): arg1(arg1), arg2(arg2)
    {

    }

    virtual void forwardPass(const Eigen::VectorXf &vars) override {
        arg1->forwardPass(vars);
        arg2->forwardPass(vars);
        auto res1 = arg1->value();
        auto res2 = arg2->value();
        mValue = res1 - res2;
    }

    virtual void backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) override {
        arg1->backwardPass( sensitivity, grad);
        arg2->backwardPass(-sensitivity, grad);
    }

    virtual const Eigen::MatrixXf &value() const override {
        return mValue;
    }

    virtual std::string toString() const override {
        return "";
    }

private:
    Eigen::MatrixXf mValue;

    GNTensorOutput *arg1;
    GNTensorOutput *arg2;
};


#endif //NUMGRIND_GNMATRIXSUB_H

