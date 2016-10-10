//
// Created by daiver on 24.09.16.
//

#include "CGMatrixSum.h"

NumGrind::CompGraph::CGMatrixSum::CGMatrixSum(NumGrind::CompGraph::CGTensorOutput *arg1,
                                              NumGrind::CompGraph::CGTensorOutput *arg2) : arg1(arg1), arg2(arg2) {

}

void NumGrind::CompGraph::CGMatrixSum::forwardPass(const Eigen::VectorXf &vars) {
    arg1->forwardPass(vars);
    arg2->forwardPass(vars);
    auto res1 = arg1->value();
    auto res2 = arg2->value();
    if (res1.rows() == res2.rows() && res1.cols() == res2.cols()) {
        this->mValue = res1.array() + res2.array();
    } else if (res1.cols() == res2.cols() && res2.rows() == 1) {
        mValue = res1;
        for (int i = 0; i < res1.rows(); ++i)
            mValue.row(i) += res2;
    }
}

void NumGrind::CompGraph::CGMatrixSum::backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) {
    auto res1 = arg1->value();
    auto res2 = arg2->value();
    if (res1.rows() == res2.rows() && res1.cols() == res2.cols()) {
        this->arg1->backwardPass(sensitivity, grad);
        this->arg2->backwardPass(sensitivity, grad);
    } else if (res1.cols() == res2.cols() && res2.rows() == 1) {
        this->arg1->backwardPass(sensitivity, grad);
        this->arg2->backwardPass(sensitivity.colwise().sum(), grad);
    }
}
