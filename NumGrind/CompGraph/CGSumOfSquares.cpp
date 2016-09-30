//
// Created by daiver on 30.09.16.
//

#include "CGSumOfSquares.h"

NumGrind::CompGraph::CGSumOfSquares::CGSumOfSquares(NumGrind::CompGraph::CGTensorOutput *arg): arg(arg)
{

}

float NumGrind::CompGraph::CGSumOfSquares::value() const {
    return mValue;
}

void NumGrind::CompGraph::CGSumOfSquares::forwardPass(const Eigen::VectorXf &vars)
{
    arg->forwardPass(vars);
    auto res = arg->value();
    this->mValue = (res.array() * res.array()).sum();
}

void NumGrind::CompGraph::CGSumOfSquares::backwardPass(const float sensitivity, Eigen::VectorXf &grad)
{
    auto res = arg->value();
    arg->backwardPass(2*res*sensitivity, grad);
}
