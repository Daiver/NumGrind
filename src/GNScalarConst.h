#ifndef NUMGRINDTEST01_GRAPHNODESCALARCONST_H
#define NUMGRINDTEST01_GRAPHNODESCALARCONST_H

#include "GNScalarOutput.h"

class GNScalarConst : public GNScalarOutput
{
public:
    GNScalarConst(const float value): mValue(value)
    {
    }

    void forwardPass(const Eigen::VectorXf &vars) {}

    float value() const { return this->mValue; }

    virtual void backwardPass(const float sensitivity, Eigen::VectorXf &grad) override {}

    virtual std::string toString() const
    {
        return "[" + std::to_string(mValue) + "]";
    }

private:
    float mValue;
};


#endif //NUMGRINDTEST01_GRAPHNODESCALARCONST_H
