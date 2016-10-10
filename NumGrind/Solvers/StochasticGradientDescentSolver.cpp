#include "StochasticGradientDescentSolver.h"

#include <iostream>

NumGrind::Solvers::StochasticGradientDescentSolver::StochasticGradientDescentSolver(
        const NumGrind::Solvers::SolverSettings &settings, const float stepSize): settings(settings), stepSize(stepSize)
{

}

void NumGrind::Solvers::StochasticGradientDescentSolver::makeStep(std::function<float(const Eigen::VectorXf &)> func,
                                                                  std::function<void(const Eigen::VectorXf &,
                                                                                     Eigen::VectorXf &)> gradFunc,
                                                                  Eigen::VectorXf &vars)
{
    this->grad.resize(vars.size());
    grad.fill(0);
    gradFunc(vars, grad);
    vars += -stepSize * grad;
    if(this->settings.verbose)
        std::cout << "err " << func(vars) << std::endl;
}
