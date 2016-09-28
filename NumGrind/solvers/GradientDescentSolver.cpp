#include "GradientDescentSolver.h"

#include <iostream>

using namespace NumGrind;

void solvers::gradientDescent(
        const SolverSettings &settings,
        const float stepSize,
        std::function<float(const Eigen::VectorXf &)> func,
        std::function<void(const Eigen::VectorXf &, Eigen::VectorXf &)> gradF, Eigen::VectorXf &vars) {
    Eigen::VectorXf grad = Eigen::VectorXf::Zero(vars.size());
    const float err = func(vars);
    std::cout << "Initial err " << err << std::endl;
    for (int iter = 0; iter < settings.nMaxIterations; ++iter) {
        gradF(vars, grad);
        vars -= grad * stepSize;
        const float err = func(vars);
        if(settings.verbose)
            std::cout << iter << " : " << err << std::endl;

        if(grad.norm() < settings.minGradL2)
            break;
    }
}
