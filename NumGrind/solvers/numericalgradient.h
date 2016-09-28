#ifndef NUMGRIND_NUMERICALGRADIENT_H
#define NUMGRIND_NUMERICALGRADIENT_H

#include <functional>
#include "Eigen/Core"

namespace solvers {
    void numericalGradient(std::function<float(const Eigen::VectorXf &)> func, const Eigen::VectorXf &varsInit,
                               const float dx, Eigen::VectorXf &grad);
}

#endif //NUMGRIND_NUMERICALGRADIENT_H
