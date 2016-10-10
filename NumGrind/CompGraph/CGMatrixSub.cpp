//
// Created by daiver on 24.09.16.
//

#include "CGMatrixSub.h"

NumGrind::CompGraph::CGMatrixSub::CGMatrixSub(NumGrind::CompGraph::CGTensorOutput *arg1,
                                              NumGrind::CompGraph::CGTensorOutput *arg2) : arg1(arg1), arg2(arg2) {

}

void NumGrind::CompGraph::CGMatrixSub::forwardPass(const Eigen::VectorXf &vars) {
    arg1->forwardPass(vars);
    arg2->forwardPass(vars);
    auto res1 = arg1->value();
    auto res2 = arg2->value();
    mValue = res1 - res2;
}

void NumGrind::CompGraph::CGMatrixSub::backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) {
    arg1->backwardPass(sensitivity, grad);
    arg2->backwardPass(-sensitivity, grad);
}
