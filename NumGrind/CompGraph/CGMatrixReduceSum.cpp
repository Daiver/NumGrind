//
// Created by daiver on 24.09.16.
//

#include "CGMatrixReduceSum.h"

NumGrind::CompGraph::CGMatrixReduceSum::CGMatrixReduceSum(NumGrind::CompGraph::CGTensorOutput *arg) : arg(arg) {

}

void NumGrind::CompGraph::CGMatrixReduceSum::forwardPass(const Eigen::VectorXf &vars) {
    arg->forwardPass(vars);
    auto res = arg->value();

    this->mValue = 0;
    for (int i = 0; i < res.rows(); ++i)
        for (int j = 0; j < res.cols(); ++j)
            this->mValue += res(i, j);
}

void NumGrind::CompGraph::CGMatrixReduceSum::backwardPass(const float sensitivity, Eigen::VectorXf &grad) {
    const auto res = arg->value();
    const int nRows = res.rows();
    const int nCols = res.cols();
    const auto sens = Eigen::MatrixXf::Constant(nRows, nCols, sensitivity);
    this->arg->backwardPass(sens, grad);
}
