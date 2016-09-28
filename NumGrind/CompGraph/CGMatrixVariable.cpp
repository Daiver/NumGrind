#include "CGMatrixVariable.h"

void NumGrind::CompGraph::CGMatrixVariable::setValue(const Eigen::MatrixXf &value)
{
    this->mNRows = value.rows();
    this->mNCols = value.cols();
    this->mValue = value;
}
