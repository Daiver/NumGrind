//
// Created by daiver on 24.09.16.
//

#include "CGScalarConst.h"

NumGrind::CompGraph::CGScalarConst::CGScalarConst(const float value) : mValue(value) {
}

void NumGrind::CompGraph::CGScalarConst::forwardPass(const Eigen::VectorXf &vars) {}

void NumGrind::CompGraph::CGScalarConst::backwardPass(const float sensitivity, Eigen::VectorXf &grad) {}
