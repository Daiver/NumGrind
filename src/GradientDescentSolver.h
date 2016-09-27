#ifndef NUMGRIND_GRADIENTDESCENTSOLVER_H
#define NUMGRIND_GRADIENTDESCENTSOLVER_H

#include "CompGraph/GNScalarOutput.h"
#include "Eigen/Core"

namespace NumGrind {
    namespace solvers {
        void gradientDescent(const int nIters, const float stepSize, CompGraph::GNScalarOutput &function, Eigen::VectorXf &vars);

    }
}
#endif //NUMGRIND_GRADIENTDESCENTSOLVER_H
