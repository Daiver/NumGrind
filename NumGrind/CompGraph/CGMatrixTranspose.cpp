//
// Created by daiver on 24.09.16.
//

#include "CGMatrixTranspose.h"

NumGrind::CompGraph::CGMatrixTranspose::CGMatrixTranspose(NumGrind::CompGraph::CGTensorOutput *arg) : arg(arg) {
    assert(false);//Not implemented yet
}

void NumGrind::CompGraph::CGMatrixTranspose::forwardPass(const Eigen::VectorXf &vars) {
    arg->forwardPass(vars);
    this->mValue = arg->value().transpose();
}

void NumGrind::CompGraph::CGMatrixTranspose::backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) {

}
