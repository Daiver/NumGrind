//
// Created by daiver on 24.09.16.
//

#include "CGMatrixScalarSum.h"

NumGrind::CompGraph::CGMatrixScalarSum::CGMatrixScalarSum(NumGrind::CompGraph::CGTensorOutput *arg1,
                                                          NumGrind::CompGraph::CGScalarOutput *arg2) : arg1(arg1), arg2(arg2) {

}

void NumGrind::CompGraph::CGMatrixScalarSum::forwardPass(const Eigen::VectorXf &vars) {
    arg1->forwardPass(vars);
    arg2->forwardPass(vars);
    mValue = arg1->value().array() + arg2->value();
}

void NumGrind::CompGraph::CGMatrixScalarSum::backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) {
    arg1->backwardPass(sensitivity, grad);
    arg2->backwardPass(sensitivity.array().sum(), grad);
}
