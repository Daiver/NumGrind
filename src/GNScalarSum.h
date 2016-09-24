#ifndef NUMGRINDTEST01_GRAPHNODEFUNCSCALARSUM_H
#define NUMGRINDTEST01_GRAPHNODEFUNCSCALARSUM_H

#include "GNScalarFunction.h"

class GNScalarSum : public GNScalarFunction
{
public:
    GNScalarSum(GNScalarOutput *argA, GNScalarOutput *argB)
    {
        this->arguments.push_back(argA);
        this->arguments.push_back(argB);
    }
    void forwardPass(const Eigen::VectorXf &vars)
    {
        float res = 0.0;
        for(GNScalarOutput *arg : this->arguments){
            arg->forwardPass(vars);
            res += arg->value();
        }
        this->mValue = res;
    }

    virtual void backwardPass(const float sensitivity, Eigen::VectorXf &grad) override
    {
        for (GNScalarOutput *arg : arguments) {
            arg->backwardPass(sensitivity, grad);
        }
    }

    virtual std::string toString() const
    {
        std::string res = "";
        for(int i = 0; i < arguments.size() - 1; ++i)
            res += arguments[i]->toString() + " + ";
        res += arguments[arguments.size() - 1]->toString();
        return "[" + res + ", " + std::to_string(this->value()) + "]";
    }
};


#endif //NUMGRINDTEST01_GRAPHNODEFUNCSCALARSUM_H
