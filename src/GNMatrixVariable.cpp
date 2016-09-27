#include "GNMatrixVariable.h"

void NumGrind::CompGraph::GNMatrixVariable::setValue(const Eigen::MatrixXf &value)
{
    this->mNRows = value.rows();
    this->mNCols = value.cols();
    this->mValue = value;
}
