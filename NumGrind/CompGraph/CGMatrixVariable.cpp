#include "CGMatrixVariable.h"

void NumGrind::CompGraph::CGMatrixVariable::setValue(const Eigen::MatrixXf &value)
{
    this->mNRows = value.rows();
    this->mNCols = value.cols();
    this->mValue = value;
}

NumGrind::CompGraph::CGMatrixVariable::CGMatrixVariable(const int nRows, const int nCols) :
        mNRows(nRows), mNCols(nCols), mValue(Eigen::MatrixXf::Zero(nRows, nCols)) {}

NumGrind::CompGraph::CGMatrixVariable::CGMatrixVariable(const int nRows, const int nCols,
                                                        const std::vector<int> &indices) :
        mNRows(nRows), mNCols(nCols), indices(indices), mValue(Eigen::MatrixXf::Zero(nRows, nCols)) {
    assert(nRows * nCols == indices.size());
}

void NumGrind::CompGraph::CGMatrixVariable::forwardPass(const Eigen::VectorXf &vars) {
    for (int i = 0; i < indices.size(); ++i) {
        auto inds2D = flatIndTo2DInd(i);
        this->mValue(inds2D.first, inds2D.second) = vars[indices[i]];
    }
}

void NumGrind::CompGraph::CGMatrixVariable::backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) {
    assert(sensitivity.rows() == mNRows);
    assert(sensitivity.cols() == mNCols);
    for (int i = 0; i < indices.size(); ++i) {
        auto inds2D = flatIndTo2DInd(i);
        grad[indices[i]] += sensitivity(inds2D.first, inds2D.second);
    }
}

std::pair<int, int> NumGrind::CompGraph::CGMatrixVariable::flatIndTo2DInd(const int flatInd) {
    const int row = flatInd / mNCols;
    const int col = flatInd % mNCols;
    return std::make_pair(row, col);
}
