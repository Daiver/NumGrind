#ifndef NUMGRINDTEST01_GRAPHNODEFUNCSCALARMULT_H
#define NUMGRINDTEST01_GRAPHNODEFUNCSCALARMULT_H

#include "GNScalarFunction.h"

class GNScalarMult : public GNScalarFunction
{
public:
    GNScalarMult(GNScalarOutput *argA, GNScalarOutput *argB)
    {
        this->arguments.push_back(argA);
        this->arguments.push_back(argB);
    }
    void forwardPass(const Eigen::VectorXf &vars)
    {
        float res = 1.0;
        for(GNScalarOutput *arg : this->arguments){
            arg->forwardPass(vars);
            res *= arg->value();
        }
        this->mValue = res;
    }

    virtual void backwardPass(const float sensitivity, Eigen::VectorXf &grad) override
    {
        assert(arguments.size() == 2);
        arguments[0]->backwardPass(sensitivity * arguments[1]->value(), grad);
        arguments[1]->backwardPass(sensitivity * arguments[0]->value(), grad);
    }

    virtual std::string toString() const
    {
        std::string res = "";
        for(int i = 0; i < arguments.size() - 1; ++i)
            res += arguments[i]->toString() + " * ";
        res += arguments[arguments.size() - 1]->toString();
        return "[" + res + ", " + std::to_string(this->value()) + "]";
    }
};


#endif //NUMGRINDTEST01_GRAPHNODEFUNCSCALARMULT_H
