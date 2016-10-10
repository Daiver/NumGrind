//
// Created by daiver on 25.09.16.
//

#include "CGScalarSub.h"

NumGrind::CompGraph::CGScalarSub::CGScalarSub(NumGrind::CompGraph::CGScalarOutput *arg1,
                                              NumGrind::CompGraph::CGScalarOutput *arg2) : arg1(arg1), arg2(arg2) {

}

void NumGrind::CompGraph::CGScalarSub::forwardPass(const Eigen::VectorXf &vars) {
    arg1->forwardPass(vars);
    arg2->forwardPass(vars);
    this->mValue = arg1->value() - arg2->value();
}

void NumGrind::CompGraph::CGScalarSub::backwardPass(const float sensitivity, Eigen::VectorXf &grad) {
    arg1->backwardPass(sensitivity, grad);
    arg2->backwardPass(-sensitivity, grad);
}
