//
// Created by daiver on 24.09.16.
//

#include "CGMatrixProduct.h"

NumGrind::CompGraph::CGMatrixProduct::CGMatrixProduct(NumGrind::CompGraph::CGTensorOutput *arg1,
                                                      NumGrind::CompGraph::CGTensorOutput *arg2) : arg1(arg1), arg2(arg2) {

}

void NumGrind::CompGraph::CGMatrixProduct::forwardPass(const Eigen::VectorXf &vars) {
    arg1->forwardPass(vars);
    arg2->forwardPass(vars);
    auto res1 = arg1->value();
    auto res2 = arg2->value();
    assert(res1.cols() == res2.rows());
    this->mValue = res1 * res2;
}

void NumGrind::CompGraph::CGMatrixProduct::backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) {
    auto res1 = arg1->value();
    auto res2 = arg2->value();

//        int res1R = res1.rows();
//        int res1C = res1.cols();
//
//        int res2R = res2.rows();
//        int res2C = res2.cols();
//
//        int sensR = sensitivity.rows();
//        int sensC = sensitivity.cols();

    arg1->backwardPass((sensitivity * res2.transpose()), grad);
    arg2->backwardPass((res1.transpose() * sensitivity), grad);
}
