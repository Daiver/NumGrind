//
// Created by daiver on 24.09.16.
//

#include "CGScalarMult.h"

NumGrind::CompGraph::CGScalarMult::CGScalarMult(NumGrind::CompGraph::CGScalarOutput *argA,
                                                NumGrind::CompGraph::CGScalarOutput *argB) {
    this->arguments.push_back(argA);
    this->arguments.push_back(argB);
}

void NumGrind::CompGraph::CGScalarMult::forwardPass(const Eigen::VectorXf &vars) {
    float res = 1.0;
    for (CGScalarOutput *arg : this->arguments) {
        arg->forwardPass(vars);
        res *= arg->value();
    }
    this->mValue = res;
}

void NumGrind::CompGraph::CGScalarMult::backwardPass(const float sensitivity, Eigen::VectorXf &grad) {
    assert(arguments.size() == 2);
    arguments[0]->backwardPass(sensitivity * arguments[1]->value(), grad);
    arguments[1]->backwardPass(sensitivity * arguments[0]->value(), grad);
}
