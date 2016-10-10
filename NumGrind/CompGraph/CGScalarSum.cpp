//
// Created by daiver on 24.09.16.
//

#include "CGScalarSum.h"

NumGrind::CompGraph::CGScalarSum::CGScalarSum(NumGrind::CompGraph::CGScalarOutput *argA,
                                              NumGrind::CompGraph::CGScalarOutput *argB) {
    this->arguments.push_back(argA);
    this->arguments.push_back(argB);
}

void NumGrind::CompGraph::CGScalarSum::forwardPass(const Eigen::VectorXf &vars) {
    float res = 0.0;
    for (CGScalarOutput *arg : this->arguments) {
        arg->forwardPass(vars);
        res += arg->value();
    }
    this->mValue = res;
}

void NumGrind::CompGraph::CGScalarSum::backwardPass(const float sensitivity, Eigen::VectorXf &grad) {
    for (CGScalarOutput *arg : arguments) {
        arg->backwardPass(sensitivity, grad);
    }
}
