#ifndef NUMGRIND_GRADIENTDESCENTSOLVER_H
#define NUMGRIND_GRADIENTDESCENTSOLVER_H

#include <functional>
#include "Eigen/Core"
#include "SolverSettings.h"

namespace NumGrind {
    namespace solvers {
        void gradientDescent(
                const SolverSettings &settings, const float stepSize,
                std::function<float(const Eigen::VectorXf &)> func,
                std::function<void(const Eigen::VectorXf &, Eigen::VectorXf &)> gradF, Eigen::VectorXf &vars);

    }
}
#endif //NUMGRIND_GRADIENTDESCENTSOLVER_H
