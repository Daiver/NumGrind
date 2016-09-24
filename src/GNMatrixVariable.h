#ifndef NUMGRINDTEST01_GNMATRIXVARIABLE_H
#define NUMGRINDTEST01_GNMATRIXVARIABLE_H

#include <algorithm>
#include "GNTensorOutput.h"

class GNMatrixVariable : public GNTensorOutput{
public:
    GNMatrixVariable(const int nRows, const int nCols, const std::vector<int> &indices):
            nRows(nRows), nCols(nCols), indices(indices), mValue(Eigen::MatrixXf::Zero(nRows, nCols))
    {
        assert(nRows * nCols == indices.size());
    }

    void forwardPass(const Eigen::VectorXf &vars)
    {
        for(int i = 0; i < indices.size(); ++i) {
            auto inds2D = flatIndTo2DInd(i);
            this->mValue(inds2D.first, inds2D.second) = vars[indices[i]];
        }
    }

    virtual void backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) override
    {
        assert(sensitivity.rows() == nRows);
        assert(sensitivity.cols() == nCols);
        for(int i = 0; i < indices.size(); ++i) {
            auto inds2D = flatIndTo2DInd(i);
            grad[indices[i]] += sensitivity(inds2D.first, inds2D.second);
        }
    }

    const Eigen::MatrixXf &value() const { return this->mValue; }

    virtual std::string toString() const
    {
        return "";
    }

protected:
    std::pair<int, int> flatIndTo2DInd(const int flatInd)
    {
        const int row = flatInd / nCols;
        const int col = flatInd % nCols;
        return std::make_pair(row, col);
    }

private:
    const int nRows;
    const int nCols;
    const std::vector<int> indices;
    Eigen::MatrixXf mValue;
};


#endif //NUMGRINDTEST01_GNMATRIXVARIABLE_H
