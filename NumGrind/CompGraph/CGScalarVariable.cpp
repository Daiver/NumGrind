//
// Created by daiver on 24.09.16.
//

#include "CGScalarVariable.h"

NumGrind::CompGraph::CGScalarVariable::CGScalarVariable(const int index) : index(index), mValue(0.0) {
}

void NumGrind::CompGraph::CGScalarVariable::forwardPass(const Eigen::VectorXf &vars) {
    assert(index < vars.size());
    this->mValue = vars[index];
}

void NumGrind::CompGraph::CGScalarVariable::backwardPass(const float sensitivity, Eigen::VectorXf &grad) {
    assert(index < grad.size());
    grad[this->index] += sensitivity;
}
