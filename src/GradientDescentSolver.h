#ifndef NUMGRIND_GRADIENTDESCENTSOLVER_H
#define NUMGRIND_GRADIENTDESCENTSOLVER_H

#include "GNScalarOutput.h"
#include "Eigen/Core"

namespace solvers{
    void gradientDescent(const int nIters, const float stepSize, GNScalarOutput &function, Eigen::VectorXf &vars);

}

#endif //NUMGRIND_GRADIENTDESCENTSOLVER_H
