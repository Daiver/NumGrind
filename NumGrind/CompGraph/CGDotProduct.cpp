//
// Created by daiver on 24.09.16.
//

#include "CGDotProduct.h"

void NumGrind::CompGraph::CGDotProduct::forwardPass(const Eigen::VectorXf &vars) {
    arg1->forwardPass(vars);
    arg2->forwardPass(vars);

    auto res = arg1->value().transpose() * arg2->value();
    assert(res.rows() == 1);
    this->mValue = res(0, 0);
}

NumGrind::CompGraph::CGDotProduct::CGDotProduct(NumGrind::CompGraph::CGTensorOutput *arg1,
                                                NumGrind::CompGraph::CGTensorOutput *arg2) : arg1(arg1), arg2(arg2) {

}

void NumGrind::CompGraph::CGDotProduct::backwardPass(const float sensitivity, Eigen::VectorXf &grad) {
    auto res1 = arg1->value();
    auto res2 = arg2->value();
    this->arg1->backwardPass(sensitivity * res2, grad);
    this->arg2->backwardPass(sensitivity * res1, grad);
}
