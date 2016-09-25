#include "GNMatrixVariable.h"

void GNMatrixVariable::setValue(const Eigen::MatrixXf &value)
{
    this->mNRows = value.rows();
    this->mNCols = value.cols();
    this->mValue = value;
}
