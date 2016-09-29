//
// Created by daiver on 29.09.16.
//

#include "SymbolicTensorConstant.h"

NumGrind::SymbolicGraph::SymbolicTensorConstant::SymbolicTensorConstant(
        NumGrind::SymbolicGraph::SymbolicGraphManagerAbstract *manager,
        NumGrind::CompGraph::CGMatrixConstant *graphNode): SymbolicTensorNode(manager, graphNode) {

}

void NumGrind::SymbolicGraph::SymbolicTensorConstant::setValue(const Eigen::VectorXf value) {
    this->mNodeConstant->setValue(value);
}
