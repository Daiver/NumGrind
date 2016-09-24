#ifndef NUMGRINDTEST01_GRAPHNODESCALARVARIABLE_H
#define NUMGRINDTEST01_GRAPHNODESCALARVARIABLE_H

#include "GNScalarOutput.h"

class GNScalarVariable : public GNScalarOutput
{
public:
    GNScalarVariable(const int index): index(index), mValue(0.0)
    {
    }

    void forwardPass(const Eigen::VectorXf &vars)
    {
        assert(index < vars.size());
        this->mValue = vars[index];
    }

    virtual void backwardPass(const float sensitivity, Eigen::VectorXf &grad) override
    {
        assert(index < grad.size());
        grad[this->index] += sensitivity;
    }

    float value() const { return this->mValue; }

    virtual std::string toString() const
    {
        return "[X" + std::to_string(index) + ":" + std::to_string(mValue) + "]";
    }

private:
    const int index;
    float mValue;
};


#endif //NUMGRINDTEST01_GRAPHNODESCALARVARIABLE_H
