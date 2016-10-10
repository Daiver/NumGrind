//
// Created by daiver on 24.09.16.
//

#include "CGMatrixElementWiseProduct.h"

NumGrind::CompGraph::CGMatrixElementWiseProduct::CGMatrixElementWiseProduct(NumGrind::CompGraph::CGTensorOutput *arg1,
                                                                            NumGrind::CompGraph::CGTensorOutput *arg2) : arg1(arg1), arg2(arg2) {

}

void NumGrind::CompGraph::CGMatrixElementWiseProduct::forwardPass(const Eigen::VectorXf &vars) {
    arg1->forwardPass(vars);
    arg2->forwardPass(vars);
    this->mValue = arg1->value().array() * arg2->value().array();
}

void
NumGrind::CompGraph::CGMatrixElementWiseProduct::backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) {
    auto res1 = arg1->value();
    auto res2 = arg2->value();
    this->arg1->backwardPass(sensitivity.array() * res2.array(), grad);
    this->arg2->backwardPass(sensitivity.array() * res1.array(), grad);
}
