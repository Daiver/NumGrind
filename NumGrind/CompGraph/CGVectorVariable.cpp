//
// Created by daiver on 24.09.16.
//

#include "CGVectorVariable.h"

NumGrind::CompGraph::CGVectorVariable::CGVectorVariable(const std::vector<int> &indices) : indices(indices),
                                                                                           mValue(Eigen::VectorXf::Zero(indices.size())) {
}

void NumGrind::CompGraph::CGVectorVariable::forwardPass(const Eigen::VectorXf &vars) {
    assert(this->mValue.cols() == 1);
    for (int i = 0; i < indices.size(); ++i)
        this->mValue(i, 0) = vars[indices[i]];
}

void NumGrind::CompGraph::CGVectorVariable::backwardPass(const Eigen::MatrixXf &sensitivity, Eigen::VectorXf &grad) {
    assert(sensitivity.cols() == 1);
    for (int i = 0; i < indices.size(); ++i)
        grad[indices[i]] += sensitivity(i, 0);
}
