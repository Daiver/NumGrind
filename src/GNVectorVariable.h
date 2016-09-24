#ifndef NUMGRINDTEST01_GRAPHNODEVECTORVARIABLE_H
#define NUMGRINDTEST01_GRAPHNODEVECTORVARIABLE_H

#include <vector>
#include "GNTensorOutput.h"

class GNVectorVariable : public GNTensorOutput
{
public:
    GNVectorVariable(const std::vector<int> &indices): indices(indices), mValue(Eigen::VectorXf::Zero(indices.size()))
    {
    }

    void forwardPass(const Eigen::VectorXf &vars)
    {
        assert(this->mValue.cols() == 1);
        for(int i = 0; i < indices.size(); ++i)
            this->mValue(i, 0) = vars[indices[i]];
    }

    virtual void backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) override
    {
        assert(sensitivity.cols() == 1);
        for(int i = 0; i < indices.size(); ++i)
            grad[indices[i]] += sensitivity(i, 0);
    }

    const Eigen::MatrixXf &value() const { return this->mValue; }

    virtual std::string toString() const
    {
        std::string res = "[ ";
        for(int i = 0; i < mValue.rows(); ++i)
            res += std::to_string(mValue(i, 0)) + " ";
        res += "]";
        return "[Vec:" + res + "]";
    }

private:
    const std::vector<int> indices;
    Eigen::MatrixXf mValue;
};


#endif //NUMGRINDTEST01_GRAPHNODEVECTORVARIABLE_H
