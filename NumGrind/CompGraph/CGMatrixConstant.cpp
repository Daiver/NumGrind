//
// Created by daiver on 24.09.16.
//

#include "CGMatrixConstant.h"

NumGrind::CompGraph::CGMatrixConstant::CGMatrixConstant(const Eigen::MatrixXf &value) : mValue(value) {

}

void NumGrind::CompGraph::CGMatrixConstant::forwardPass(const Eigen::VectorXf &vars) {}

void NumGrind::CompGraph::CGMatrixConstant::backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) {}
