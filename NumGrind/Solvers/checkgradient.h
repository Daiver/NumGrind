#ifndef NUMGRIND_CHECKGRADIENT_H
#define NUMGRIND_CHECKGRADIENT_H

#include "numericalgradient.h"

namespace NumGrind {
    namespace Solvers {
        bool isGradientOk(
                std::function<float (const Eigen::VectorXf &)> func,
                std::function<void (const Eigen::VectorXf &, Eigen::VectorXf &)> grad,
                const Eigen::VectorXf &vars,
                const float tolerance = 1e-3,
                const float dx = 1e-3);
    }
}

#endif //NUMGRIND_CHECKGRADIENT_H
